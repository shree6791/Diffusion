#!/usr/bin/env python
"""
Load LoRA-adapted U-Net into SD pipeline and generate images.
"""
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from PIL import Image

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load base Stable Diffusion v1.5
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to(device)

    # 2) Apply your LoRA weights
    lora_dir = "./lora_sks_person"
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_dir)

    # 3) Generate
    prompt = "a photo of sks_person riding a horse"
    image = pipe(
        prompt=prompt,
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]

    # 4) Save
    image.save("lora_output.jpg")
    print("Saved -> lora_output.jpg")

if __name__ == "__main__":
    main()