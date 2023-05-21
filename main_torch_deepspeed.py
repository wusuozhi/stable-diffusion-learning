import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config
from einops import rearrange
from tqdm import tqdm
from txt2img import load_model_from_config
import deepspeed
deepspeed.init_distributed()

MULTINODE_HACKS = False

@rank_zero_only
def rank_zero_print(*args):
    print(*args)

def modify_weights(w, scale = 1e-6, n=2):
    """Modify weights to accomodate concatenation to unet"""
    extra_w = scale*torch.randn_like(w)
    new_w = w.clone()
    for i in range(n):
        new_w = torch.cat((new_w, extra_w.clone()), dim=1)
    return new_w


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--finetune_from",
        type=str,
        nargs="?",
        default="weights/sd-v1-4-full-ema.ckpt",
        help="path to checkpoint to load model state from"
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=['configs/stable-diffusion/pokemon.yaml'],
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    return parser


def test_model(model,epoch):
    from ldm.models.diffusion.ddim import DDIMSampler
    sampler = DDIMSampler(model)
    sampler.fp16= fp16 #True
    prompts = ['a photograph of an astronaut riding a horse']
    batch_size=1
    uc = model.get_learned_conditioning(batch_size * [""])
    c = model.get_learned_conditioning(prompts)
    shape = [4, 512//8, 512//8]
    xt_shape = [1,4, 512//8, 512//8]
    if fp16:
        start_code = torch.randn(xt_shape, device=device).half()
    else:
        start_code = torch.randn(xt_shape, device=device)
    samples_ddim, _ = sampler.sample(S=50,
                                        conditioning=c,
                                        batch_size=batch_size,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=5,
                                        unconditional_conditioning=uc,
                                        x_T=start_code)
    if fp16:
        samples_ddim = samples_ddim.half()
    x_samples_ddim = model.decode_first_stage(samples_ddim)
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

    for i,x_sample in enumerate(x_samples_ddim):
        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')

        Image.fromarray(x_sample.astype(np.uint8)).save(f"./logs/main_torch_deepspeed/{i}_{epoch}.png")


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    fp16=True
    bs = 1 
    deepspeed_config = {
            'train_micro_batch_size_per_gpu': bs,
            "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-4,
                "eps": 1e-8,
                "weight_decay": 0
            }
            },
            'gradient_accumulation_steps': 1,
            'gradient_clipping': 4,
            'fp16': {
                'enabled': fp16,
            },
            'steps_per_print': 100000000,
            "flops_profiler": {
                "enabled": False,
                "profile_step": 1,
                "module_depth": -1,
                "top_modules": 1,
                "detailed": True,
                "output_file": None
            },
            "zero_optimization": {
                "stage": 2,
            },
        }


    # 0、init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    # 1、模型
    model = instantiate_from_config(config.model)
    print("Loading Model Success!")

    # 1.5 加载预训练模型
    # print(f"load resume {opt.finetune_from}")

    old_state = torch.load(opt.finetune_from, map_location="cpu")
    try:
        sd = old_state["state_dict"]
    except:
        sd = old_state
    m, u = model.load_state_dict(sd, strict=False)

    # 2、optimizer
    lr = 1e-4
    params = list(model.model.parameters())
    opt = torch.optim.AdamW(params, lr=lr)
    
    # 3、dataset
    dataset = instantiate_from_config(config.data.params.train)
    print("Loading data Success!")


    # 4、wrapper with  deepspeed
    model_engine, optimizer, trainloader, _ = deepspeed.initialize(args=deepspeed_config, 
                                                        model=model.model, 
                                                        optimizer=opt,
                                                        model_parameters=filter(lambda p: p.requires_grad, model.model.parameters()), 
                                                        config_params=deepspeed_config,
                                                        training_data=dataset)
    model.cuda()

    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True, num_workers=2)
    
    model.fp16=fp16
    if fp16:
        # model.model.diffusion_model.convert_to_fp16()
        model.model = model.model.half()
        model.cond_stage_model = model.cond_stage_model.half()
        model.first_stage_model = model.first_stage_model.half()
    

    print("Training!")
    # 4、Start train
    device= torch.device(model_engine.local_rank)
    for epoch in range(10*6*5): # 800/8 = 100   2*50/gpu/epoch, 300
        
        for i,bs in enumerate(tqdm(trainloader,desc=f"{epoch}")):
            if fp16:
                bs['image'] = bs['image'].cuda().half()
            else:
                bs['image'] = bs['image'].cuda()
            loss = model.training_step(bs, i)
            model_engine.backward(loss)
            model_engine.step()
            model.on_train_batch_end()
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
            torch.cuda.synchronize()
        
        test_model(model,epoch)
        saving_path = f"logs/main_torch_deepspeed/{epoch}.pt"
        
        if model_engine.local_rank==0 and epoch % 30 ==0:
            torch.save(model.state_dict(),saving_path)
        
