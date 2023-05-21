export NCCL_IB_DISABLE=1
CUDA_VISIBLE_DEVICES=6,7 NCCL_DEBUG=ERROR deepspeed main_torch_deepspeed.py
