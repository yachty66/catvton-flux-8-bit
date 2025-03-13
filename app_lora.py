import spaces

import gradio as gr
from tryon_inference import run_inference
import os
import numpy as np
from PIL import Image
import tempfile
import torch
from diffusers import FluxTransformer2DModel, FluxFillPipeline
import subprocess

subprocess.run("rm -rf /data-nvme/zerogpu-offload/*", env={}, shell=True)
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Start loading LoRA weights")
state_dict, network_alphas = FluxFillPipeline.lora_state_dict(
    pretrained_model_name_or_path_or_dict="xiaozaa/catvton-flux-lora-alpha",     ## The tryon Lora weights
    weight_name="pytorch_lora_weights.safetensors",
    return_alphas=True
)
is_correct_format = all("lora" in key or "dora_scale" in key for key in state_dict.keys())
if not is_correct_format:
    raise ValueError("Invalid LoRA checkpoint.")
print('Loading diffusion model ...')
pipe = FluxFillPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Fill-dev",
    torch_dtype=torch.bfloat16
).to(device)
FluxFillPipeline.load_lora_into_transformer(
    state_dict=state_dict,
    network_alphas=network_alphas,
    transformer=pipe.transformer,
)

print('Loading Finished!')

@spaces.GPU
def gradio_inference(
    image_data, 
    garment, 
    num_steps=50, 
    guidance_scale=30.0, 
    seed=-1, 
    width=768,
    height=1024
):
    """Wrapper function for Gradio interface"""
    # Use temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save inputs to temp directory
        temp_image = os.path.join(tmp_dir, "image.png")
        temp_mask = os.path.join(tmp_dir, "mask.png")
        temp_garment = os.path.join(tmp_dir, "garment.png")
        
        # Extract image and mask from ImageEditor data
        image = image_data["background"]
        mask = image_data["layers"][0]  # First layer contains the mask
        
        # Convert to numpy array and process mask
        mask_array = np.array(mask)
        is_black = np.all(mask_array < 10, axis=2)
        mask = Image.fromarray(((~is_black) * 255).astype(np.uint8))
        
        # Save files to temp directory
        image.save(temp_image)
        mask.save(temp_mask)
        garment.save(temp_garment)
        
        try:
            # Run inference
            _, tryon_result = run_inference(
                pipe=pipe,
                image_path=temp_image,
                mask_path=temp_mask,
                garment_path=temp_garment,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                size=(width, height)
            )
            return tryon_result
        except Exception as e:
            raise gr.Error(f"Error during inference: {str(e)}")

with gr.Blocks() as demo:
    gr.Markdown("""
    # CATVTON FLUX Virtual Try-On Demo (by using LoRA weights)
    Upload a model image, draw a mask, and a garment image to generate virtual try-on results.
    
    [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/xiaozaa/catvton-flux-alpha)
    [![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/nftblackmagic/catvton-flux)
    """)
    
    # gr.Video("example/github.mp4", label="Demo Video: How to use the tool")
    
    with gr.Column():
        with gr.Row():
            with gr.Column():
                image_input = gr.ImageMask(
                    label="Model Image (Click 'Edit' and draw mask over the clothing area)",
                    type="pil",
                    height=600,
                    width=300
                )
                gr.Examples(
                    examples=[
                        ["./example/person/00008_00.jpg"],
                        ["./example/person/00055_00.jpg"],
                        ["./example/person/00057_00.jpg"],
                        ["./example/person/00067_00.jpg"],
                        ["./example/person/00069_00.jpg"],
                    ],
                    inputs=[image_input],
                    label="Person Images",
                )
            with gr.Column():
                garment_input = gr.Image(label="Garment Image", type="pil", height=600, width=300)
                gr.Examples(
                    examples=[
                        ["./example/garment/04564_00.jpg"],
                        ["./example/garment/00055_00.jpg"],
                        ["./example/garment/00396_00.jpg"],
                        ["./example/garment/00067_00.jpg"],
                        ["./example/garment/00069_00.jpg"],
                    ],
                    inputs=[garment_input],
                    label="Garment Images",
                ) 
            with gr.Column():
                tryon_output = gr.Image(label="Try-On Result", height=600, width=300)
            
        with gr.Row():
            num_steps = gr.Slider(
                minimum=1, 
                maximum=100, 
                value=30, 
                step=1, 
                label="Number of Steps"
            )
            guidance_scale = gr.Slider(
                minimum=1.0, 
                maximum=50.0, 
                value=30.0, 
                step=0.5, 
                label="Guidance Scale"
            )
            seed = gr.Slider(
                minimum=-1,
                maximum=2147483647,
                step=1,
                value=-1,
                label="Seed (-1 for random)"
            )
            width = gr.Slider(
                minimum=256,
                maximum=1024,
                step=64,
                value=768,
                label="Width"
            )
            height = gr.Slider(
                minimum=256,
                maximum=1024,
                step=64,
                value=1024,
                label="Height"
            )
            
        
        submit_btn = gr.Button("Generate Try-On", variant="primary")
        
            
    with gr.Row():
        gr.Markdown("""
        ### Notes:
        - The model is trained on VITON-HD dataset. It focuses on the woman upper body try-on generation.
        - The mask should indicate the region where the garment will be placed.
        - The garment image should be on a clean background.
        - The model is not perfect. It may generate some artifacts.
        - The model is slow. Please be patient.
        - The model is just for research purpose.
        """)
    
    submit_btn.click(
        fn=gradio_inference,
        inputs=[
            image_input,
            garment_input,
            num_steps,
            guidance_scale,
            seed,
            width,
            height
        ],
        outputs=[tryon_output],
        api_name="try-on"
    )


demo.launch()
