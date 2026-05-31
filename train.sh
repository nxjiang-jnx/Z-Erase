export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
export HF_HUB_OFFLINE=1

python train_ZImage_lora.py --config config/config.yaml