from argparse import ArgumentParser
from os.path import basename, join, isdir
from pathlib import Path
import pandas as pd
from glob import glob
import numpy as np
import select
import sys

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy

from datamodule import VapDataModule

from callbacks import SymmetricSpeakersCallback
from train import VAPModel, DataConfig, OptConfig, get_run_name
# from vap.phrases.dataset import PhrasesCallback
from utils import everything_deterministic, write_json

# Delete later prolly
from model import VapGPT, VapConfig
from events import TurnTakingEvents, EventConfig
# from vap.zero_shot import ZeroShot

import datetime

everything_deterministic()

MIN_THRESH = 0.01  # Minimum `threshold` limit for S/L, S-pred, BC-pred
ROOT = "runs_evaluation"


def get_args():
    parser = ArgumentParser("VoiceActivityProjection")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.ckpt",
    )
    # parser = pl.Trainer.add_argparse_args(parser)
    # parser = OptConfig.add_argparse_args(parser)
    parser = DataConfig.add_argparse_args(parser)
    parser, fields_added = VapConfig.add_argparse_args(parser)
    parser, fields_added = EventConfig.add_argparse_args(parser, fields_added)
    parser.add_argument("--devices", type=str, default='0')
    args = parser.parse_args()

    model_conf = VapConfig.args_to_conf(args)
    # opt_conf = OptConfig.args_to_conf(args)
    data_conf = DataConfig.args_to_conf(args)
    event_conf = EventConfig.args_to_conf(args)

    # Remove all non trainer args
    cfg_dict = vars(args)
    for k, _ in list(cfg_dict.items()):
        if (
            k.startswith("data_")
            or k.startswith("vap_")
            or k.startswith("opt_")
            or k.startswith("event_")
        ):
            cfg_dict.pop(k)

    return {
        "args": args,
        "cfg_dict": cfg_dict,
        "model": model_conf,
        "event": event_conf,
        # "opt": opt_conf,
        "data": data_conf,
    }

def get_savepath(args, configs):
    name = basename(args.checkpoint).replace(".ckpt", "")
    # name += "_" + "_".join(configs["data"].datasets)
    savepath = join(ROOT, name)
    Path(savepath).mkdir(exist_ok=True, parents=True)
    print("SAVEPATH: ", savepath)
    # write_json(cfg_dict, join(savepath, "config.json"))
    return savepath


def evaluate() -> None:
    """Evaluate model"""

    configs = get_args()
    run_name = get_run_name(configs)
    # print(run_name)
    # input()
    args = configs["args"]
    checkpoint = args.checkpoint

    # checkpoint がディレクトリであれば、その中からvalidation lossが最小のものを選択する
    if isdir(args.checkpoint):
        print("Checkpoint is a directory. Searching for best checkpoint...")
        checkpoints = glob(join(args.checkpoint, run_name + "-epoch*.ckpt"))
        if len(checkpoints) == 0:
            raise ValueError("No checkpoints found in directory")
        checkpoint = checkpoints[0]
        min_val_loss = 1E+10
        min_epoch = -1
        for c in checkpoints:
            val_loss = float(c.split('val_')[1].split('.ckpt')[0])
            epoch = int(c.split('epoch')[1].split('-val')[0])
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_epoch = epoch
                checkpoint = c
            elif val_loss == min_val_loss:
                if epoch < min_epoch:
                    min_epoch = epoch
                    checkpoint = c
        # checkpointsのリストを表示
        print("Checkpoints: ")
        for c in checkpoints:
            print(c)
        print("Selected checkpoint: ", checkpoint)
        # print('Press Enter to continue (in 10 seconds): ')
        # i, o, e = select.select( [sys.stdin], [], [], 10 )
    else:
        checkpoint = args.checkpoint

    cfg_dict = configs["cfg_dict"]
    savepath = get_savepath(args, configs)

    if configs["args"].devices is None:
        gpu_devices = -1
    else:
        gpu_devices = [int(d.strip()) for d in configs["args"].devices.split(",")]
    
    # if configs["args"].auto_select_gpus is not None and configs["args"].auto_select_gpus == 1:
    #     auto_select_gpus = True
    #     gpu_devices = int(configs["args"].devices)
    # else:
    #     auto_select_gpus = False
    
    #########################################################
    # Load model
    #########################################################
    model = VAPModel.load_from_checkpoint(checkpoint)
    model.eval()

    model.event_conf = configs["event"]
    model.event_extractor = TurnTakingEvents(model.event_conf)
    model.test_perturbation = configs["data"].test_perturbation

    #########################################################
    # Load data
    #########################################################
    dconf = configs["data"]
    dm = VapDataModule(
        train_path=None,
        val_path=None,
        test_path=dconf.test_path,
        horizon=2,
        batch_size=dconf.batch_size,
        num_workers=dconf.num_workers,
    )
    dm.prepare_data()
    dm.setup("test")

    #########################################################
    # Score
    #########################################################
    for pop in ["checkpoint", "seed"]:#, "gpus"]:
        cfg_dict.pop(pop)
    
    if torch.cuda.is_available():
        cfg_dict["accelerator"] = "gpu"
        cfg_dict["devices"] = gpu_devices
    else:
        cfg_dict["accelerator"] = "cpu"
        cfg_dict["devices"] = 1
        
    # print(gpu_devices)
    # if -1 not in gpu_devices:
    #     cfg_dict["accelerator"] = "gpu"
    #     cfg_dict["devices"] = gpu_devices
    #     #cfg_dict["auto_select_gpus"] = auto_select_gpus
    # else:
    #     cfg_dict["accelerator"] = "cpu"
    #     cfg_dict["devices"] = 1
    
    cfg_dict["deterministic"] = True
    cfg_dict["strategy"] = DDPStrategy(find_unused_parameters=False)

    # dm_th = VapDataModule(
    #     train_path=dconf.train_path,
    #     val_path=dconf.val_path,
    #     test_path=dconf.test_path,
    #     horizon=2,
    #     batch_size=dconf.batch_size,
    #     num_workers=dconf.num_workers,
    # )
    # dm_th.prepare_data()
    # dm_th.setup("fit")

    # thresholds = find_threshold(
    #     model, dm_th.val_dataloader(), savepath=savepath, min_thresh=MIN_THRESH, cfg_dict=cfg_dict
    # )
    # print(thresholds)

    trainer = pl.Trainer(
        callbacks=[
            SymmetricSpeakersCallback(),
            #PhrasesCallback()
        ], **cfg_dict
    )

    with torch.no_grad():
        result = trainer.test(model, dataloaders=dm.test_dataloader())[0]

    # fixup results
    flat = {}
    for k, v in result.items():
        new_name = k.replace("test_", "")
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[f"{new_name}_{kk}"] = vv.cpu().item()
        else:
            flat[new_name] = v
    df = pd.DataFrame([flat])

    name = "score"
    # if cfg_dict["precision"] == 16:
    #     name += "_fp16"
    # if cfg_dict["limit_test_batches"] is not None:
    #     nn = cfg_dict["limit_test_batches"] * dm.batch_size
    #     name += f"_nb-{nn}"

    filepath = join(savepath, name + ".csv")
    df.to_csv(filepath, index=False)
    print("Saved to -> ", filepath)
    
    # print("Added to -> ", filepath)
    

if __name__ == "__main__":
    evaluate()
