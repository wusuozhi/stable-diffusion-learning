export NCCL_IB_DISABLE=1
CUDA_VISIBLE_DEVICES=5,6 NCCL_DEBUG=ERROR deepspeed main_torch_deepspeed.py
# CUDA_VISIBLE_DEVICES=6 NCCL_DEBUG=ERROR deepspeed main_torch_deepspeed.py
