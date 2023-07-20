import torch
import torch.nn.functional as F
import PIL
import numpy as np

from typing import Any, Callable, Dict, List, Optional, Union

from transformers import CLIPImageProcessor, T5EncoderModel, T5Tokenizer
from diffusers.models import UNet2DConditionModel
from diffusers.schedulers import DDPMScheduler
from diffusers import IFPipeline, IFSuperResolutionPipeline
from diffusers.pipelines.deepfloyd_if import IFPipelineOutput
from diffusers.pipelines.deepfloyd_if.safety_checker import IFSafetyChecker
from diffusers.pipelines.deepfloyd_if.watermark import IFWatermarker
from diffusers.utils import replace_example_docstring, randn_tensor

EXAMPLE_DOC_STRING = "Sorry, No Example is available."

class IFAmbigramPipeline(IFPipeline):
    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        unet: UNet2DConditionModel,
        scheduler: DDPMScheduler,
        safety_checker: Optional[IFSafetyChecker],
        feature_extractor: Optional[CLIPImageProcessor],
        watermarker: Optional[IFWatermarker],
        requires_safety_checker: bool = True,
    ):
        super().__init__(tokenizer, text_encoder, unet, scheduler, safety_checker, feature_extractor, watermarker, requires_safety_checker)

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt_embeds_1:Dict[str, torch.FloatTensor] = None,
        prompt_embeds_2:Dict[str, torch.FloatTensor] = None,
        num_inference_steps: int = 100,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        num_images_per_prompt: Optional[int] = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        clean_caption: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        ambigram_flip = None
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt_embeds_1 ((`dict`, `torch.FloatTensor`)):
                Pre-generated text embeddings for orginal direction view. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
            prompt_embeds_2 ((`dict`, `torch.FloatTensor`)):
                Pre-generated text embeddings for fliped/rotated direction view. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process. If not defined, equal spaced `num_inference_steps`
                timesteps are used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            height (`int`, *optional*, defaults to self.unet.config.sample_size):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size):
                The width in pixels of the generated image.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            clean_caption (`bool`, *optional*, defaults to `True`):
                Whether or not to clean the caption before creating embeddings. Requires `beautifulsoup4` and `ftfy` to
                be installed. If the dependencies are not installed, the embeddings will be created from the raw
                prompt.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            ambigram_flip (Ambigramflip):
                Ambigram type. Check the detail in ambigram.py.

        Examples:

        Returns:
            Generarted images from original direction view and fliped/rotated direction view. 
            Image formats are torch Tensor or PIL Image. 
        """
        # 1. Check inputs. Raise error if not correct
        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        
        if prompt_embeds_1.get('pos', None) is None or prompt_embeds_1.get('pos', None) is None or prompt_embeds_2.get('pos', None) is None or prompt_embeds_2.get('neg', None) is None:
            raise ValueError(
                f"`prompt_embeds` and `negative_prompt_embeds` must be provided as a dictionary with keys `pos` and `neg`."
            )

        if not (prompt_embeds_1['pos'].shape == prompt_embeds_1['neg'].shape == prompt_embeds_2['pos'].shape == prompt_embeds_2['neg'].shape):
            raise ValueError(
                "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly."
            )
        
        if ambigram_flip is None:
            raise ValueError(
                "`ambigram_flip` must be provided."
            )

        # 2. Define call parameters
        height = height or self.unet.config.sample_size
        width = width or self.unet.config.sample_size
        batch_size = prompt_embeds_1['pos'].shape[0]
        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds_1['pos'], prompt_embeds_1['neg'] = self.encode_prompt(
            None,
            do_classifier_free_guidance,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            negative_prompt=None,
            prompt_embeds=prompt_embeds_1['pos'],
            negative_prompt_embeds=prompt_embeds_1['neg'],
            clean_caption=clean_caption,
        )

        if do_classifier_free_guidance:
            prompt_embeds_1 = torch.cat([prompt_embeds_1['neg'], prompt_embeds_1['pos']])

        prompt_embeds_2['pos'], prompt_embeds_2['neg'] = self.encode_prompt(
            None,
            do_classifier_free_guidance,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            negative_prompt=None,
            prompt_embeds=prompt_embeds_2['pos'],
            negative_prompt_embeds=prompt_embeds_2['neg'],
            clean_caption=clean_caption,
        )

        if do_classifier_free_guidance:
            prompt_embeds_2 = torch.cat([prompt_embeds_2['neg'], prompt_embeds_2['pos']])

        # 4. Prepare timesteps
        if timesteps is not None:
            self.scheduler.set_timesteps(timesteps=timesteps, device=device)
            timesteps = self.scheduler.timesteps
            num_inference_steps = len(timesteps)
        else:
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

        # 5. Prepare intermediate images
        intermediate_images = self.prepare_intermediate_images(
            batch_size * num_images_per_prompt,
            self.unet.config.in_channels,
            height,
            width,
            prompt_embeds_1.dtype,
            device,
            generator,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # HACK: see comment in `enable_model_cpu_offload`
        if hasattr(self, "text_encoder_offload_hook") and self.text_encoder_offload_hook is not None:
            self.text_encoder_offload_hook.offload()

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                ## make fliped intermediate_images
                fliped_intermediate_images = ambigram_flip(intermediate_images.clone(), reverse=False)

                ## org side noise_pred
                model_input = (
                    torch.cat([intermediate_images] * 2) if do_classifier_free_guidance else intermediate_images
                )
                model_input = self.scheduler.scale_model_input(model_input, t)

                noise_pred = self.unet(
                    model_input,
                    t,
                    encoder_hidden_states=prompt_embeds_1,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
                    noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1], dim=1)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

                if self.scheduler.config.variance_type not in ["learned", "learned_range"]:
                    noise_pred, _ = noise_pred.split(model_input.shape[1], dim=1)

                org_noise_pred = noise_pred.clone()

                ## fliped side noise_pred
                model_input = (
                    torch.cat([fliped_intermediate_images] * 2) if do_classifier_free_guidance else fliped_intermediate_images
                )
                model_input = self.scheduler.scale_model_input(model_input, t)

                noise_pred = self.unet(
                    model_input,
                    t,
                    encoder_hidden_states=prompt_embeds_2,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
                    noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1], dim=1)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

                if self.scheduler.config.variance_type not in ["learned", "learned_range"]:
                    noise_pred, _ = noise_pred.split(model_input.shape[1], dim=1)
                
                fliped_noise_pred = ambigram_flip(noise_pred.clone(), reverse=True) 

                # compute the previous noisy sample x_t -> x_t-1
                final_noise_pred = 0.5*org_noise_pred + 0.5*fliped_noise_pred
                intermediate_images = self.scheduler.step(
                    final_noise_pred, t, intermediate_images, **extra_step_kwargs, return_dict=False
                )[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, intermediate_images)

        image = intermediate_images
        fliped_image = ambigram_flip(intermediate_images.clone(), reverse=False)

        # 8. Post-processing
        nsfw_detected, watermark_detected = [None, None], [None, None]
        if output_type == "pil":
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = self.numpy_to_pil(image)
            fliped_image = (fliped_image / 2 + 0.5).clamp(0, 1)
            fliped_image = fliped_image.cpu().permute(0, 2, 3, 1).float().numpy()
            fliped_image = self.numpy_to_pil(fliped_image)
            
            image, nsfw_detected[0], watermark_detected[0] = self.run_safety_checker(image, device, prompt_embeds.dtype)
            fliped_image, nsfw_detected[1], watermark_detected[1] = self.run_safety_checker(fliped_image, device, prompt_embeds.dtype)

        elif output_type == "pt":
            if hasattr(self, "unet_offload_hook") and self.unet_offload_hook is not None:
                self.unet_offload_hook.offload()

        else:
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            fliped_image = (fliped_image / 2 + 0.5).clamp(0, 1)
            fliped_image = fliped_image.cpu().permute(0, 2, 3, 1).float().numpy()

            image, nsfw_detected[0], watermark_detected[0] = self.run_safety_checker(image, device, prompt_embeds.dtype)
            fliped_image, nsfw_detected[1], watermark_detected[1] = self.run_safety_checker(fliped_image, device, prompt_embeds.dtype)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if output_type == "pt":
            return image, fliped_image
        else:
            return image, fliped_image, (nsfw_detected, watermark_detected)


class IFAmbigramSuperResolutionPipeline(IFSuperResolutionPipeline):
    def __init__(
        self, 
        tokenizer: T5Tokenizer, 
        text_encoder: T5EncoderModel, 
        unet: UNet2DConditionModel, 
        scheduler: DDPMScheduler, 
        image_noising_scheduler: DDPMScheduler, 
        safety_checker: Optional[IFSafetyChecker] = None, 
        feature_extractor: Optional[CLIPImageProcessor] = None, 
        watermarker: Optional[IFWatermarker] = None, 
        requires_safety_checker: bool = True
    ):
        super().__init__(
            tokenizer, 
            text_encoder, 
            unet, scheduler, 
            image_noising_scheduler, 
            safety_checker, 
            feature_extractor, 
            watermarker, 
            requires_safety_checker
        )

    def check_inputs(
        self,
        callback_steps,
        prompt_embeds_1,
        prompt_embeds_2,
        ambigram_flip,
        noise_level,
        image,
    ):
        
        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        

        if prompt_embeds_1.get('pos', None) is None or prompt_embeds_1.get('pos', None) is None or prompt_embeds_2.get('pos', None) is None or prompt_embeds_2.get('neg', None) is None:
            raise ValueError(
                f"`prompt_embeds` and `negative_prompt_embeds` must be provided as a dictionary with keys `pos` and `neg`."
            )

        if not (prompt_embeds_1['pos'].shape == prompt_embeds_1['neg'].shape == prompt_embeds_2['pos'].shape == prompt_embeds_2['neg'].shape):
            raise ValueError(
                "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly."
            )
        
        if ambigram_flip is None:
            raise ValueError(
                "`ambigram_flip` must be provided."
            )

        if noise_level < 0 or noise_level >= self.image_noising_scheduler.config.num_train_timesteps:
            raise ValueError(
                f"`noise_level`: {noise_level} must be a valid timestep in `self.noising_scheduler`, [0, {self.image_noising_scheduler.config.num_train_timesteps})"
            )

        if isinstance(image, list):
            check_image_type = image[0]
        else:
            check_image_type = image

        if (
            not isinstance(check_image_type, torch.Tensor)
            and not isinstance(check_image_type, PIL.Image.Image)
            and not isinstance(check_image_type, np.ndarray)
        ):
            raise ValueError(
                "`image` has to be of type `torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, or List[...] but is"
                f" {type(check_image_type)}"
            )

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt_embeds_1:Dict[str, torch.FloatTensor] = None,
        prompt_embeds_2:Dict[str, torch.FloatTensor] = None,
        height: int = None,
        width: int = None,
        image: Union[PIL.Image.Image, np.ndarray, torch.FloatTensor] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 4.0,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        noise_level: int = 250,
        clean_caption: bool = True,
        ambigram_flip = None
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt_embeds_1 ((`dict`, `torch.FloatTensor`)):
                Pre-generated text embeddings for orginal direction view. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
            prompt_embeds_2 ((`dict`, `torch.FloatTensor`)):
                Pre-generated text embeddings for fliped/rotated direction view. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
            height (`int`, *optional*, defaults to self.unet.config.sample_size):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size):
                The width in pixels of the generated image.
            image (`PIL.Image.Image`, `np.ndarray`, `torch.FloatTensor`):
                The image to be upscaled.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process. If not defined, equal spaced `num_inference_steps`
                timesteps are used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            noise_level (`int`, *optional*, defaults to 250):
                The amount of noise to add to the upscaled image. Must be in the range `[0, 1000)`
            clean_caption (`bool`, *optional*, defaults to `True`):
                Whether or not to clean the caption before creating embeddings. Requires `beautifulsoup4` and `ftfy` to
                be installed. If the dependencies are not installed, the embeddings will be created from the raw
                prompt.
            ambigram_flip (Ambigramflip):
                Ambigram type. Check the detail in ambigram.py.

        Examples:

        Returns:
            Generarted images from original direction view and fliped/rotated direction view. 
            Image formats are torch Tensor or PIL Image.
        """
        # 1. Check inputs. Raise error if not correct
        batch_size = prompt_embeds_1['pos'].shape[0]
        
        self.check_inputs(
            callback_steps,
            prompt_embeds_1,
            prompt_embeds_2,
            ambigram_flip,
            noise_level,
            image,
        )

        # 2. Define call parameters
        height = height or self.unet.config.sample_size
        width = width or self.unet.config.sample_size

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds_1['pos'], prompt_embeds_1['neg'] = self.encode_prompt(
            None,
            do_classifier_free_guidance,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            negative_prompt=None,
            prompt_embeds=prompt_embeds_1['pos'],
            negative_prompt_embeds=prompt_embeds_1['neg'],
            clean_caption=clean_caption,
        )

        if do_classifier_free_guidance:
            prompt_embeds_1 = torch.cat([prompt_embeds_1['neg'], prompt_embeds_1['pos']])

        prompt_embeds_2['pos'], prompt_embeds_2['neg'] = self.encode_prompt(
            None,
            do_classifier_free_guidance,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            negative_prompt=None,
            prompt_embeds=prompt_embeds_2['pos'],
            negative_prompt_embeds=prompt_embeds_2['neg'],
            clean_caption=clean_caption,
        )

        if do_classifier_free_guidance:
            prompt_embeds_2 = torch.cat([prompt_embeds_2['neg'], prompt_embeds_2['pos']])

        # 4. Prepare timesteps
        if timesteps is not None:
            self.scheduler.set_timesteps(timesteps=timesteps, device=device)
            timesteps = self.scheduler.timesteps
            num_inference_steps = len(timesteps)
        else:
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

        # 5. Prepare intermediate images
        num_channels = self.unet.config.in_channels // 2
        intermediate_images = self.prepare_intermediate_images(
            batch_size * num_images_per_prompt,
            num_channels,
            height,
            width,
            prompt_embeds_1.dtype,
            device,
            generator,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare upscaled image and noise level
        image = self.preprocess_image(image, num_images_per_prompt, device)
        upscaled = F.interpolate(image, (height, width), mode="bilinear", align_corners=True)

        noise_level = torch.tensor([noise_level] * upscaled.shape[0], device=upscaled.device)
        noise = randn_tensor(upscaled.shape, generator=generator, device=upscaled.device, dtype=upscaled.dtype)
        upscaled = self.image_noising_scheduler.add_noise(upscaled, noise, timesteps=noise_level)
        fliped_upscaled = ambigram_flip(upscaled.clone(), reverse=False)

        if do_classifier_free_guidance:
            noise_level = torch.cat([noise_level] * 2)

        # HACK: see comment in `enable_model_cpu_offload`
        if hasattr(self, "text_encoder_offload_hook") and self.text_encoder_offload_hook is not None:
            self.text_encoder_offload_hook.offload()

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                ## make fliped intermediate_images
                fliped_intermediate_images = ambigram_flip(intermediate_images.clone(), reverse=False)

                ## org side noise_pred
                model_input = torch.cat([intermediate_images, upscaled], dim=1)
                model_input = torch.cat([model_input] * 2) if do_classifier_free_guidance else model_input
                model_input = self.scheduler.scale_model_input(model_input, t)

                noise_pred = self.unet(
                    model_input,
                    t,
                    encoder_hidden_states=prompt_embeds_1,
                    class_labels=noise_level,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1] // 2, dim=1)
                    noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1] // 2, dim=1)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

                if self.scheduler.config.variance_type not in ["learned", "learned_range"]:
                    noise_pred, _ = noise_pred.split(intermediate_images.shape[1], dim=1)

                org_noise_pred = noise_pred.clone()

                ## fliped side noise_pred
                model_input = torch.cat([fliped_intermediate_images, fliped_upscaled], dim=1)
                model_input = torch.cat([model_input] * 2) if do_classifier_free_guidance else model_input
                model_input = self.scheduler.scale_model_input(model_input, t)

                noise_pred = self.unet(
                    model_input,
                    t,
                    encoder_hidden_states=prompt_embeds_2,
                    class_labels=noise_level,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1] // 2, dim=1)
                    noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1] // 2, dim=1)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

                if self.scheduler.config.variance_type not in ["learned", "learned_range"]:
                    noise_pred, _ = noise_pred.split(intermediate_images.shape[1], dim=1)
                
                fliped_noise_pred = ambigram_flip(noise_pred.clone(), reverse=True) 

                # compute the previous noisy sample x_t -> x_t-1
                final_noise_pred = 0.5*org_noise_pred + 0.5*fliped_noise_pred
                intermediate_images = self.scheduler.step(
                    final_noise_pred, t, intermediate_images, **extra_step_kwargs, return_dict=False
                )[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, intermediate_images)

        image = intermediate_images
        fliped_image = ambigram_flip(intermediate_images.clone(), reverse=False)

        # 8. Post-processing
        nsfw_detected, watermark_detected = [None, None], [None, None]
        if output_type == "pil":
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image, nsfw_detected[0], watermark_detected[0] = self.run_safety_checker(image, device, prompt_embeds.dtype)
            image = self.numpy_to_pil(image)

            fliped_image = (fliped_image / 2 + 0.5).clamp(0, 1)
            fliped_image = fliped_image.cpu().permute(0, 2, 3, 1).float().numpy()
            fliped_image, nsfw_detected[1], watermark_detected[1] = self.run_safety_checker(fliped_image, device, prompt_embeds.dtype)
            fliped_image = self.numpy_to_pil(fliped_image)

        elif output_type == "pt":
            if hasattr(self, "unet_offload_hook") and self.unet_offload_hook is not None:
                self.unet_offload_hook.offload()

        else:
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image, nsfw_detected[0], watermark_detected[0] = self.run_safety_checker(image, device, prompt_embeds.dtype)

            fliped_image = (fliped_image / 2 + 0.5).clamp(0, 1)
            fliped_image = fliped_image.cpu().permute(0, 2, 3, 1).float().numpy()
            fliped_image, nsfw_detected[1], watermark_detected[1] = self.run_safety_checker(fliped_image, device, prompt_embeds.dtype)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if output_type == "pt":
            return image, fliped_image
        else:
            return image, fliped_image, (nsfw_detected, watermark_detected)
