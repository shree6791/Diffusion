import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from ip_adapter import IPAdapter
from PIL import Image

# 1) Load and configure Stable Diffusion v1.5
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to(device)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# 2) Instantiate IP-Adapter
ip_adapter = IPAdapter(
    pipe,
    image_encoder_path="models/image_encoder",    # dir with config.json + pytorch_model.bin
    ip_ckpt="models/ip-adapter_sd15.bin",         # SD1.5 adapter checkpoint
    device=device
)

# 3) Helper to build image grids

def image_grid(imgs, rows, cols, pad=2):
    w, h = imgs[0].size
    grid = Image.new("RGB", (cols*w + pad*(cols-1), rows*h + pad*(rows-1)), "white")
    for idx, img in enumerate(imgs):
        r, c = divmod(idx, cols)
        grid.paste(img, (c*(w+pad), r*(h+pad)))
    return grid

# 4) Generate styled variations
ref = Image.open("common_man.jpeg").convert("RGB").resize((512, 512))
images = ip_adapter.generate(
    pil_image=ref,
    prompt="a superhero flying through the sky",
    num_samples=2,
    num_inference_steps=50,
    seed=42,
    guidance_scale=7.5
)

# 5) Create and save the grid
grid = image_grid(images, rows=1, cols=4)
grid.save("common_man_variation_grid.jpg", quality=90)

# Display inline
display(grid)