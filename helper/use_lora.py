#!/usr/bin/env python
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from PIL import Image

# Monkey-patch get_default_device for compatibility with Transformers ≥4.40
if not hasattr(torch, "get_default_device"):
    torch.get_default_device = lambda: (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

def main():
    device = torch.get_default_device()

    # 1️⃣ Load base Stable Diffusion v1.5 in fp16
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
    ).to(device)

    # 2️⃣ Inject your fine-tuned LoRA adapter
    lora_dir = "./lora_sks_person"
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_dir).to(device)

    # (optional) re-enable safety checker if you need it in production:
    # pipe.safety_checker = your_safety_checker_fn

    # 3️⃣ Generate!
    prompt = "a photo of woman having coffee"
    output = pipe(
        prompt=prompt,
        num_inference_steps=100,
        guidance_scale=7.5,
        height=512,
        width=512,
    )
    image = output.images[0]

    # 4️⃣ Save out
    image.save("lora_output.jpg", quality=90)
    print("✨ Saved → lora_output.jpg")

if __name__ == "__main__":
    main()