Prerequisites

- GPU-enabled environment (e.g., Google Colab, Kaggle) with CUDA support.
- Git LFS installed for handling large model checkpoints.
- Python 3.8+.IP Adapter

Setup

1. Clone the IP-Adapter Code

```shell
# Ensure we start in /content (Colab default)
%cd /content

# Remove any previous clones\!rm -rf IP-Adapter

# Clone the official IP-Adapter repository
!git clone https://github.com/tencent-ailab/IP-Adapter.git
```shell

2. Install the IP-Adapter Python Package

```shell
# Enter the cloned directory
%cd /content/IP-Adapter

# Install in editable mode with demo extras
!pip install -e .[demo]

# Return to the root directory
%cd /content
```

This will install:

ip-adapter package and the inference entrypoint.

All demo dependencies: diffusers, transformers, opencv-python, gradio, safetensors, etc.


3. Download Model Weights

```shell
# Remove any stale folders\!rm -rf models sdxl_models IP-Adapter-models

# Initialize Git LFS and clone the Hugging Face LFS repo of weights
!git lfs install
!git clone https://huggingface.co/h94/IP-Adapter.git /content/IP-Adapter-models

# Move checkpoints into the expected locations
!mv /content/IP-Adapter-models/models ./models
!mv /content/IP-Adapter-models/sdxl_models ./sdxl_models

# Clean up intermediate folder
!rm -rf /content/IP-Adapter-models
```

Now you have:

- models/ with SD1.5 adapter weights + image encoder.

- sdxl_models/ with SDXL adapter weights + image encoder.
