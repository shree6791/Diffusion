from diffusers import StableDiffusionPipeline
import torch
from torchvision import transforms
from PIL import Image
from IPython.display import display
import matplotlib.pyplot as plt

# ✅ Load Stable Diffusion v1.5 in float16 for GPU efficiency
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# ✅ Set scheduler to run 50 steps
pipe.scheduler.set_timesteps(num_inference_steps=50)

# ✅ Prompt + CFG setup
prompt = "a fox astronaut in a neon forest"
guidance_scale = 7.5

# ✅ Encode prompt (conditional)
cond_input = pipe.tokenizer(
    prompt,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=pipe.tokenizer.model_max_length
).input_ids.to("cuda")

with torch.no_grad():
    cond = pipe.text_encoder(cond_input)[0].to(dtype=torch.float16)

# ✅ Encode empty prompt (unconditional)
uncond_input = pipe.tokenizer(
    "",
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=pipe.tokenizer.model_max_length
).input_ids.to("cuda")

with torch.no_grad():
    uncond = pipe.text_encoder(uncond_input)[0].to(dtype=torch.float16)

# ✅ Initialize random latent noise
latents = torch.randn((1, 4, 64, 64), device="cuda", dtype=torch.float16)

# ✅ Prepare transform and snapshot collector
to_pil = transforms.ToPILImage()
frames = []

# ✅ Denoising loop with CFG
for i, t in enumerate(pipe.scheduler.timesteps):
    with torch.no_grad():
        noise_uncond = pipe.unet(latents, t, encoder_hidden_states=uncond).sample
        noise_cond   = pipe.unet(latents, t, encoder_hidden_states=cond).sample
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # Save 6 intermediate frames
        if i % (50 // 6) == 0 or i == len(pipe.scheduler.timesteps) - 1:
            scaled_latents = latents * (1 / 0.18215)
            decoded = pipe.vae.decode(scaled_latents).sample
            image = to_pil(decoded[0].clamp(0, 1).cpu())
            frames.append(image)

# ✅ Show final image
display(frames[-1])

# ✅ Plot snapshots across denoising
fig, axs = plt.subplots(1, len(frames), figsize=(3 * len(frames), 4))
for i, ax in enumerate(axs):
    ax.imshow(frames[i])
    ax.axis("off")
    ax.set_title(f"Step {i * (50 // len(frames))}")
plt.tight_layout()
plt.show()