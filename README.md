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

# 3、数据
* 1) pokemon原始数据
链接: https://pan.baidu.com/s/1GDfax9DN4FP1UcHVqfgOlw?pwd=8w2g 提取码: 8w2g 
下载后将 .zip 文件放在 <项目目录>/data/ 下并解压，得到 <项目目录>/data/pokemon_data/ 目录

* 2) 翻译数据
链接: https://pan.baidu.com/s/1fK8wywhfB-IMYiacU5iBrQ?pwd=wwpp 提取码: wwpp
