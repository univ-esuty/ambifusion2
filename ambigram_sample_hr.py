import sys, os
import torch
import torchvision
import safetensors.torch as safetensors

from dataclasses import dataclass, asdict
from omegaconf import OmegaConf

from ambigram_pipeline.pipeline_if_ambigram import IFAmbigramPipeline, IFAmbigramSuperResolutionPipeline
from ambigram.ambigram import Ambigramflip

@dataclass
class TestConfigs:
    prompt_1:str = "An image of a dog."
    prompt_2:str = "An image of a cat."
    ambigram_flip:str = "rot_180"
    guidance_scale:float = 5
    batch_size:int = 8
    num_inference_steps:int = 100
    save_path:str = "./results/hr_ambigram_dog-cat"

if __name__ == '__main__':
    test_configs = TestConfigs()
    OmegaConf.save(OmegaConf.create(asdict(test_configs)), f"{test_configs.save_path}.yaml")

    ## stage 1
    stage_1 = IFAmbigramPipeline.from_pretrained(
        "DeepFloyd/IF-I-XL-v1.0", 
        torch_dtype=torch.float32,
        watermarker=None,
    )
    stage_1.enable_model_cpu_offload()
    # stage_1.scheduler = DDIMScheduler.from_config(stage_1.scheduler.config)

    ## stage 2
    stage_2 = IFAmbigramSuperResolutionPipeline.from_pretrained(
        "DeepFloyd/IF-II-L-v1.0", 
        torch_dtype=torch.float32,
        text_encoder=None, 
        watermarker=None,
    )
    stage_2.enable_model_cpu_offload()
    # stage_2.scheduler = DDIMScheduler.from_config(stage_2.scheduler.config)

    ## text embeds
    prompt_1 = [test_configs.prompt_1] * test_configs.batch_size
    prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt_1)
    prompt_embeds_1 = {'pos':prompt_embeds, 'neg':negative_embeds}

    prompt_2 = [test_configs.prompt_2] * test_configs.batch_size
    prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt_2)
    prompt_embeds_2 = {'pos':prompt_embeds, 'neg':negative_embeds}

    generator = torch.manual_seed(0)
    result_tensor_1, fliped_result_tensor_1 = stage_1(
        prompt_embeds_1=prompt_embeds_1,
        prompt_embeds_2=prompt_embeds_2,
        generator=generator, 
        guidance_scale=test_configs.guidance_scale,
        num_inference_steps=test_configs.num_inference_steps,
        output_type="pt",
        ambigram_flip=Ambigramflip(test_configs.ambigram_flip),
    )

    result_tensor_2, fliped_result_tensor_2 = stage_2(
        image=result_tensor_1.clone(),
        prompt_embeds_1=prompt_embeds_1,
        prompt_embeds_2=prompt_embeds_2,
        generator=generator, 
        guidance_scale=test_configs.guidance_scale,
        num_inference_steps=test_configs.num_inference_steps,
        output_type="pt",
        ambigram_flip=Ambigramflip(test_configs.ambigram_flip),
    )

    torchvision.utils.save_image(result_tensor_2, f'{test_configs.save_path}.png')
    torchvision.utils.save_image(fliped_result_tensor_2, f'{test_configs.save_path}_fliped.png')