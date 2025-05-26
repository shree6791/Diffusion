# # 1) Clone & install IP-Adapter
# %cd /content
# !git clone https://github.com/tencent-ailab/IP-Adapter.git
# %cd IP-Adapter
# !pip install -e .[demo]

# # 2) Pull down model weights
# %cd /content
# !git lfs install
# !git clone https://huggingface.co/h94/IP-Adapter.git IP-Adapter-models
# !mv IP-Adapter-models/models .
# !mv IP-Adapter-models/sdxl_models .

!pip install \
    torch==2.2.1+cu118 \
    torchvision==0.17.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

!pip install \
    diffusers>=0.33.1 \
    transformers>=4.40.0 \
    accelerate>=0.22.0 \
    huggingface-hub>=0.23.0 \
    peft \
    datasets \
    pillow \
    opencv-python \
    safetensors

!python helper/train_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --instance_data_dir="./training_images" \
  --output_dir="./lora_sks_person" \
  --instance_prompt="a photo of sks_person" \
  --use_8bit_adam \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --train_batch_size=1 \
  --max_train_steps=400

!python helper/use_lora.py