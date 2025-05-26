#!/usr/bin/env python
"""
Fine-tune a Stable Diffusion U-Net with a LoRA adapter.
"""

import os
from pathlib import Path
import argparse

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

# Monkey-patch get_default_device for compatibility with Transformers ≥4.40
if not hasattr(torch, "get_default_device"):
    torch.get_default_device = lambda: (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.schedulers import DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, get_scheduler
from peft import get_peft_model, LoraConfig
from accelerate import Accelerator


class FlatImageDataset(Dataset):
    """
    Recursively loads *all* image files under `folder` into a flat list.
    """
    def __init__(self, folder: str, transform=None):
        exts = ("jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff")
        self.paths = []
        for ext in exts:
            self.paths += list(Path(folder).rglob(f"*.{ext}"))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--instance_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--instance_prompt", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=400)
    parser.add_argument("--use_8bit_adam", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator()

    # 1️⃣ Tokenizer & text encoder
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = text_encoder.to(accelerator.device)

    # 2️⃣ Stable Diffusion pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        safety_checker=None
    )
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(accelerator.device)

    # 3️⃣ LoRA wrap of UNet
    unet = pipe.unet
    unet.config.model_type = "unet2dcondition"
    lora_config = LoraConfig(
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_config)
    pipe.unet = unet

    # 4️⃣ Dataset & transforms
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    dataset = FlatImageDataset(
        folder=os.path.join(args.instance_data_dir),
        transform=preprocess
    )

    def collate_fn(batch):
        images = torch.stack(batch, dim=0)             # [B,3,512,512]
        prompts = [args.instance_prompt] * images.size(0)
        tokenized = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        # move tokens to device
        tokenized = {k: v.to(accelerator.device) for k, v in tokenized.items()}
        return {
            "pixel_values": images,
            **tokenized
        }

    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # 5️⃣ Optimizer & LR scheduler
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.max_train_steps
    )

    # 6️⃣ Prepare with Accelerate
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )

    # 7️⃣ Training loop
    global_step = 0
    for _, batch in enumerate(dataloader):
        with accelerator.accumulate(unet):
            pixel_values = batch["pixel_values"]
            # ensure batch and 3 channels
            if pixel_values.ndim == 3:
                pixel_values = pixel_values.unsqueeze(0)
            if pixel_values.shape[1] == 1:
                pixel_values = pixel_values.repeat(1, 3, 1, 1)
            # to device & dtype
            pixel_values = pixel_values.to(accelerator.device).to(pipe.vae.dtype)
            pixel_values = pixel_values * 2.0 - 1.0

            # VAE encode + noise
            latents = pipe.vae.encode(pixel_values).latent_dist.sample()
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                pipe.scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=latents.device,
            )
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            # 🆕 Text embeddings with attention mask
            encoder_outputs = text_encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            encoder_hidden_states = encoder_outputs.last_hidden_state.to(pipe.unet.dtype)

            # UNet prediction & loss
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = torch.nn.functional.mse_loss(model_pred, noise)
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        global_step += 1
        if global_step >= args.max_train_steps:
            break

    # 8️⃣ Save LoRA weights
    accelerator.wait_for_everyone()
    unet.save_pretrained(args.output_dir, safe_serialization=True)
    print(f"LoRA weights saved to {args.output_dir}")


if __name__ == "__main__":
    main()
