"""
This demo is based on ambigram_sample_diffusers.py
"""
import random
import torch
import safetensors.torch as safetensors
from dataclasses import dataclass, asdict
from omegaconf import OmegaConf
from PIL import Image

from ambigram_pipeline.pipeline_if_ambigram import IFAmbigramPipeline
from ambigram.ambigram import Ambigramflip

from ambigram_sample import TestConfigs

test_configs = TestConfigs()

def main(
    input_c1,
    input_c2,
    ambigram_type,
    class_scale,
    batch_size,
    num_inference_steps,
    seed,
):
    ## Convert input
    _input_c1 = f'{input_c1}'
    _input_c2 = f'{input_c2}'
    _ambigram_type = ambigram_type.lower()
    _class_scale = float(class_scale)
    _batch_size = int(batch_size)
    _num_inference_steps = int(num_inference_steps)
    _seed = int(seed)

    stage_1 = IFAmbigramPipeline.from_pretrained(
        "DeepFloyd/IF-I-XL-v1.0", 
        torch_dtype=torch.float32,
        watermarker=None,
    )
    stage_1.enable_model_cpu_offload()

    prompt_1 = [_input_c1] * _batch_size
    prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt_1)
    prompt_embeds_1 = {'pos':prompt_embeds, 'neg':negative_embeds}

    prompt_2 = [_input_c2] * _batch_size
    prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt_2)
    prompt_embeds_2 = {'pos':prompt_embeds, 'neg':negative_embeds}

    generator = torch.manual_seed(_seed)
    result_imgs, fliped_imgs = stage_1(
        prompt_embeds_1=prompt_embeds_1,
        prompt_embeds_2=prompt_embeds_2,
        generator=generator, 
        guidance_scale=_class_scale,
        num_inference_steps=_num_inference_steps,
        output_type="pil",
        ambigram_flip=Ambigramflip(_ambigram_type),
    )

    output_list = []
    output_list += [result_imgs]
    output_list += [fliped_imgs]
    
    return output_list

import gradio as gr
with gr.Blocks() as demo:
    gr.HTML("<h1 align='center' style='color:yellow;'>Ambigram Generation by Diffusers [DEMO]</h1>")

    with gr.Column(variant="panel"):
        gr.Markdown(
        """
        ## Enter a pair of prompt to generate ambigram.
        (e.g.) c1="An image of letter 'A'", c2="An image of letter 'Y'" ⇒ it generates 'A↕Y'  
        """
        )
        with gr.Row(variant="compact"):
            input_c1 = gr.Textbox(label="INPUT_C1", show_label=False, max_lines=1, placeholder="Enter prompt [C1]",)
            input_c2 = gr.Textbox(label="INPUT_C2", show_label=False, max_lines=1, placeholder="Enter prompt [C2]",)
            btn = gr.Button("Generate image")

        with gr.Row(variant="compact"):
            ambigram_type = gr.Radio(["rot_180", "rot_p90", "rot_n90", "LR_flip", "UD_flip", "identity"], value="rot_180", label="Ambigram Type", info="chose ambigram type.")
            class_scale = gr.Slider(1.0, 20.0, value=5.0, label="classifier-free gudiance scale", info="", step=0.5, interactive=True)
 
        with gr.Row(variant="compact"):
            batch_size = gr.Slider(1, 16, value=4, label="batch size", info="", step=1, interactive=True)
            num_inference_steps = gr.Slider(20, 100, value=50, label="num inference steps", info="", step=5, interactive=True)
            seed = gr.Slider(0, 65535, value=random.randint(0, 65535), label="manual seed", info="", step=1, interactive=True)

    with gr.Column(variant="panel"):
        gr.Markdown(
        """
        ## Generated ambigrams [from C1 direction]
        """
        )
        gallery_1 = gr.Gallery(
            label="gallery_1", show_label=False, elem_id="gallery_1", columns=12, rows=3, object_fit="contain", height="auto"
        )
        gr.Markdown(
        """
        ## Generated ambigrams [from C2 direction]
        """
        )
        gallery_2 = gr.Gallery(
            label="gallery_2", show_label=False, elem_id="gallery_2", columns=12, rows=3, object_fit="contain", height="auto"
        )
        
    btn.click(
        main, 
        inputs=[
            input_c1,
            input_c2,
            ambigram_type,
            class_scale,
            batch_size,
            num_inference_steps,
            seed,
        ], 
        outputs=[
            gallery_1, 
            gallery_2, 
        ])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=11111)