
"""

# 1) Clone & install IP-Adapter
# %cd /content
# !git clone https://github.com/tencent-ailab/IP-Adapter.git
# %cd IP-Adapter
# !pip install -e .[demo]

# 2) Pull down model weights
# %cd /content
# !git lfs install
# !git clone https://huggingface.co/h94/IP-Adapter.git IP-Adapter-models
# !mv IP-Adapter-models/models .
# !mv IP-Adapter-models/sdxl_models .

"""

!pip install --upgrade pip

!pip install \
    diffusers>=0.33.1 \
    transformers>=4.40.0 \
    accelerate>=0.22.0 \
    huggingface-hub>=0.23.0 \
    peft \
    datasets \
    pillow \
    opencv-python \
    safetensors \
    hf_xet

import numpy, torch, torchvision
import diffusers, transformers, accelerate, peft

print("numpy:", numpy.__version__)     # should be <2, e.g. 1.25.x
print("torch:", torch.__version__, torch.version.cuda)  # 2.2.1+cu118
print("torchvision:", torchvision.__version__)          # 0.17.1+cu118
print("diffusers:", diffusers.__version__)
print("transformers:", transformers.__version__)
print("accelerate:", accelerate.__version__)

"""**LORA**"""

!python train_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --instance_data_dir="./training_images" \
  --output_dir="./lora_santorini_style" \
  --instance_prompt="a photo of santorini_style_dress" \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --train_batch_size=1 \
  --max_train_steps=500 \
  --use_8bit_adam

!python use_lora.py

"""**IP Adapter**"""

import sys
sys.path.append("/content/IP-Adapter")

import gc
torch.cuda.empty_cache()
gc.collect()

#!/usr/bin/env python
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    DPMSolverMultistepScheduler,
    ControlNetModel,
    # enable_model_cpu_offload
)
from peft import PeftModel
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL = "runwayml/stable-diffusion-v1-5"
LORA_DIR   = "./lora_santorini_style"
IP_ENCODER = "models/image_encoder"              # your SD1.5 encoder folder
IP_CKPT    = "models/ip-adapter_sd15.bin"        # your SD1.5 IP-Adapter checkpoint

CTRL_CANNY  = "lllyasviel/sd-controlnet-canny"
CTRL_POSE   = "lllyasviel/sd-controlnet-openpose"
CTRL_SCRIB  = "lllyasviel/sd-controlnet-scribble"

# reference images on disk
REF_STYLE   = "./training_images/sks_santorini_dress/santorini_dress_reference/santorini_dress_reference_white.jpeg"
CN_CANNY    = REF_STYLE #"./training_images/controlnet_default/canny_img.jpeg"
CN_POSE     = "./training_images/controlnet_default/openpose.jpeg"
CN_SCRIBBLE = "./training_images/controlnet_default/color_hint.jpeg"

PROMPT          = "A flowy Santorini-style maxi dress"
NEGATIVE_PROMPT = "blurry, bad anatomy"
NUM_VARIATIONS = 4
STEPS           = 250
GUIDANCE        = 7.5
HEIGHT, WIDTH   = 512, 512

# ─────────────────────────────────────────────────────────────────────────────
#  1) Load core Stable Diffusion ONCE
# ─────────────────────────────────────────────────────────────────────────────
pipe_core = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float16
).to(DEVICE)

# VRAM & speed savers
pipe_core.enable_attention_slicing()
try:
    pipe_core.enable_xformers_memory_efficient_attention()
except:
    pass

# Offload rarely-used modules to CPU
# enable_model_cpu_offload(pipe_core, DEVICE)

# Extract core pieces
vae, text_enc, tokenizer, unet, sched, feat_ext = (
    pipe_core.vae,
    pipe_core.text_encoder,
    pipe_core.tokenizer,
    pipe_core.unet,
    pipe_core.scheduler,
    pipe_core.feature_extractor,
)

# ─────────────────────────────────────────────────────────────────────────────
#  2) Apply LoRA
# ─────────────────────────────────────────────────────────────────────────────
unet = PeftModel.from_pretrained(unet, LORA_DIR).to(DEVICE)

# ─────────────────────────────────────────────────────────────────────────────
#  3) Wrap with IP-Adapter
# ─────────────────────────────────────────────────────────────────────────────
from ip_adapter import IPAdapter
ip_pipe = pipe_core.__class__( # clone same class to hold adapters
    vae=vae, text_encoder=text_enc, tokenizer=tokenizer,
    unet=unet, scheduler=sched, safety_checker=pipe_core.safety_checker,
    feature_extractor=feat_ext
).to(DEVICE)

ip_pipe = IPAdapter(
    ip_pipe,
    image_encoder_path=IP_ENCODER,
    ip_ckpt=IP_CKPT,
    device=DEVICE
)

# ─────────────────────────────────────────────────────────────────────────────
#  4) Build ControlNet pipeline
# ─────────────────────────────────────────────────────────────────────────────
# load 3 controlnets
cn1 = ControlNetModel.from_pretrained(CTRL_CANNY,  torch_dtype=torch.float16).to(DEVICE)
cn2 = ControlNetModel.from_pretrained(CTRL_POSE,   torch_dtype=torch.float16).to(DEVICE)
# cn3 = ControlNetModel.from_pretrained(CTRL_SCRIB,  torch_dtype=torch.float16).to(DEVICE)

ctrl_pipe = StableDiffusionControlNetPipeline(
    vae=vae,
    text_encoder=text_enc,
    tokenizer=tokenizer,
    unet=ip_pipe.pipe.unet,        # Re-use the IP-Adapter-wrapped UNet
    controlnet=[cn1, cn2],
    # controlnet=[cn1, cn2, cn3],
    scheduler=DPMSolverMultistepScheduler.from_config(sched.config),
    feature_extractor=feat_ext,
    safety_checker=pipe_core.safety_checker,
).to(DEVICE)

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers: load & preprocess reference images
# ─────────────────────────────────────────────────────────────────────────────
def load_ref(path):
    return Image.open(path).convert("RGB").resize((HEIGHT, WIDTH))

ref_style   = load_ref(REF_STYLE)
cn_canny    = load_ref(CN_CANNY)
cn_pose     = load_ref(CN_POSE)
cn_scribble = load_ref(CN_SCRIBBLE)

# ─────────────────────────────────────────────────────────────────────────────
#  5) Generate stage 1: style via IP-Adapter
# ─────────────────────────────────────────────────────────────────────────────
stage1_imgs = ip_pipe.generate(
    pil_image=ref_style,              # your 512×512 reference
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    num_inference_steps=STEPS,
    guidance_scale=GUIDANCE,
    num_samples=NUM_VARIATIONS,
)

# pick one variant to structure-control in stage 2:
seed_img = stage1_imgs[0]

for i, img in enumerate(stage1_imgs):
    img.save(f"stage1_{i}.png")

"""#### **Control Net**"""

# ─────────────────────────────────────────────────────────────────────────────
#  6) Generate stage 2: structure via ControlNet
# ─────────────────────────────────────────────────────────────────────────────
final_imgs = ctrl_pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    num_inference_steps=STEPS,
    guidance_scale=GUIDANCE,
    image=[cn_canny, cn_pose],    # list matches the 3 ControlNets
    # image=[cn_canny, cn_pose, cn_scribble],    # list matches the 3 ControlNets
    conditioning_image=[seed_img],             # single IP output
    num_images_per_prompt=NUM_VARIATIONS,
).images

# ─────────────────────────────────────────────────────────────────────────────
#  7) Save out
# ─────────────────────────────────────────────────────────────────────────────
for i, img in enumerate(final_imgs):
    img.save(f"mvp_output_{i}.jpg", quality=90)
print("✅ Done! Generated", len(final_imgs), "images as mvp_output_*.jpg")



"""Refrence Images"""

# run this in a notebook cell (no extra download required beyond pip installs below)

# 1) install dependencies
!pip install opencv-python mediapipe pillow matplotlib

# 2) load & preprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw
import mediapipe as mp
import matplotlib.pyplot as plt

# path to your reference dress photo
REF_PATH = "./training_images/sks_santorini_dress/sks_santorini_dress_03.jpeg"

# load & resize
orig = cv2.imread(REF_PATH)
orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
h, w = orig.shape[:2]
scale = 512 / max(h, w)
orig_rs = cv2.resize(orig, (int(w*scale), int(h*scale)))
canvas = np.zeros((512,512,3), dtype=np.uint8)
y0, x0 = (512 - orig_rs.shape[0])//2, (512 - orig_rs.shape[1])//2
canvas[y0:y0+orig_rs.shape[0], x0:x0+orig_rs.shape[1]] = orig_rs

# 3) Canny edges
gray  = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
edges = cv2.Canny(gray, 100, 200)
canny_img = Image.fromarray(edges).convert("RGB")

# 4) OpenPose‐style keypoints via MediaPipe
mp_pose = mp.solutions.pose
pose   = mp_pose.Pose(static_image_mode=True)
results = pose.process(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
pose_pose = Image.new("RGB", (512,512), "white")
draw = ImageDraw.Draw(pose_pose)
if results.pose_landmarks:
    for lm in results.pose_landmarks.landmark:
        x_px = int(lm.x * 512)
        y_px = int(lm.y * 512)
        draw.ellipse((x_px-3, y_px-3, x_px+3, y_px+3), fill="black")

# 5) Color-block scribble:
#    here we just flood-fill the main dress area.
#    For a true color hint you’d hand‐paint regions—but here’s a quick automated mask:
mask = (gray > 30)  # crude: everything but background
scribble = Image.new("RGB", (512,512), (255,255,255))
scribble_arr = np.array(scribble)
scribble_arr[mask] = [200, 150,  50]  # e.g. a warm gold block
scribble = Image.fromarray(scribble_arr)

# 6) show ’em
fig, axes = plt.subplots(1,4, figsize=(16,4))
for ax, img, title in zip(axes,
    [Image.fromarray(canvas), canny_img, pose_pose, scribble],
    ["reference","canny","pose","scribble"]):
    ax.imshow(img); ax.set_title(title); ax.axis("off")
plt.show()

# now save to disk if you like:
canny_img.save("canny_img.jpeg")
pose_pose.save("pose_map.jpeg")
scribble.save("color_hint.jpeg")

