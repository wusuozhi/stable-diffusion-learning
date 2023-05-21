# 1、说明
本项目是对 https://github.com/justinpinkney/stable-diffusion 项目的改造

# 2、启动方式

原始pytorch-lighting
```
python main.py -t --base configs/stable-diffusion/pokemon.yaml --gpus 6,7  --scale_lr False --num_nodes 1  --check_val_every_n_epoch 10 --finetune_from weights/sd-v1-4-full-ema.ckpt
```

torch
```
python3 main_torch.py
```

deepspeed
```
sh sd_run.sh
```