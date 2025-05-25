!pip install controlnet_aux

import matplotlib.pyplot as plt
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux import (
    OpenposeDetector, MLSDdetector, HEDdetector, CannyDetector,
    MidasDetector, LineartAnimeDetector, NormalBaeDetector, SamDetector
)
import torch

# ─── Pipeline builder ─────────────────────────────────────────────────────────
def get_controlnet_pipeline(model_name):
    controlnet = ControlNetModel.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16
    ).to("cuda")
    return pipe

# ─── Detectors + config ──────────────────────────────────────────────────────
detectors = {
    "OpenPose": (
        OpenposeDetector.from_pretrained("lllyasviel/Annotators"),
        "openpose.jpg",
        "lllyasviel/sd-controlnet-openpose",
        "a robot meditating like a monk",
        "Extracts human body keypoints for pose control"
    ),
    "Depth": (
        MidasDetector.from_pretrained("lllyasviel/Annotators"),
        "depth.jpg",
        "lllyasviel/sd-controlnet-depth",
        "a futuristic city overgrown with vines",
        "Estimates per-pixel depth for 3D-aware draping"
    ),
    "Canny": (
        CannyDetector(),
        "canny.jpg",
        "lllyasviel/control_v11p_sd15_canny",
        "a sci-fi throne made of crystal and steel",
        "Detects image edges for sharp structural guidance"
    ),
    "Scribble": (
        HEDdetector.from_pretrained("lllyasviel/Annotators"),
        "scribble.jpg",
        "lllyasviel/control_v11p_sd15_scribble",
        "a Disney-style bunny king",
        "Extracts contour/scribble lines for freeform edits"
    ),
    "MLSD": (
        MLSDdetector.from_pretrained("lllyasviel/Annotators"),
        "mlsd.jpg",
        "lllyasviel/sd-controlnet-mlsd",
        "blueprints of a futuristic spacecraft",
        "Extracts line segments for technical detailing"
    ),
    "Segmentation": (
        SamDetector.from_pretrained("ybelkada/segment-anything", subfolder="checkpoints"),
        "segmentation.jpg",
        "lllyasviel/control_v11p_sd15_seg",
        "a floating village with glowing trees",
        "Generates semantic masks to isolate garments"
    ),
    "NormalMap": (
        NormalBaeDetector.from_pretrained("lllyasviel/Annotators"),
        "normal_map.jpg",
        "lllyasviel/sd-controlnet-normal",
        "an ancient temple carved into a mountain",
        "Computes surface normals for texture wrapping"
    ),
    "LineartAnime": (
        LineartAnimeDetector.from_pretrained("lllyasviel/Annotators"),
        "lineart_anime.jpg",
        "lllyasviel/control_v11p_sd15_lineart",
        "a Ghibli-style anime girl",
        "Detects clean anime-style outlines"
    ),
    "Tile": (
        None,
        "tile.jpg",
        "lllyasviel/control_v11f1e_sd15_tile",
        "a geometric tileable wallpaper",
        "Uses img2img for seamless, repeatable patterns"
    ),
    "QRCode": (
        None,
        "qr_code.jpg",
        "monster-labs/control_v1p_sd15_qrcode_monster",
        "a cyberpunk poster with a QR code",
        "Conditions on QR codes for functional art generation"
    )
}

# ─── Preload all pipelines just once ──────────────────────────────────────────
pipelines = {}
for _, (_, _, model_name, _, _) in detectors.items():
    if model_name not in pipelines:
        pipelines[model_name] = get_controlnet_pipeline(model_name)

# ─── Generate & visualize ────────────────────────────────────────────────────
results = {}
for category, (detector, infile, model_name, prompt, description) in detectors.items():
    pipe = pipelines[model_name]
    img = Image.open(infile).convert("RGB")

    # for all except Tile, run mask→T2I
    if category != "Tile":
        cond = detector(img) if detector else img
        output = pipe(prompt=prompt, image=cond, num_inference_steps=20).images[0]
    else:
        # if you need Tile as img2img, you can similarly preload an Img2Img pipeline
        output = pipe(prompt=prompt, image=img, num_inference_steps=20).images[0]

    results[category] = (img, output, description)

for category, (orig, proc, description) in results.items():
    prompt = detectors[category][3]

    # create figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(8, 5))

    # draw a 3-line title with different font sizes
    fig.text(0.5, 1.05, category, ha='center', va='center', fontsize=14, weight='bold')
    fig.text(0.5, 0.98, description, ha='center', va='center', fontsize=12)
    fig.text(0.5, 0.92, f'Prompt: "{prompt}"', ha='center', va='center', fontsize=10)

    # plot input/output and add a light border
    for ax, image, title in zip(axes, (orig, proc), ("Input", "Output")):
        ax.imshow(image)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")
            spine.set_linewidth(1)

    # Save image
    fig.savefig(f"{category}_comparison.png", dpi=300, bbox_inches="tight")

    # leave room at top for the title texts
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.show()

detectors.keys()

