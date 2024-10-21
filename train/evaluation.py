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


def get_curves(preds, target, pos_label=1, thresholds=None, EPS=1e-6):
    """
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)

    """

    if thresholds is None:
        thresholds = torch.linspace(0, 1, steps=101)

    if pos_label == 0:
        raise NotImplementedError("Have not done this")

    ba, f1 = [], []
    auc0, auc1 = [], []
    prec0, rec0 = [], []
    prec1, rec1 = [], []
    pos_label_idx = torch.where(target == 1)
    neg_label_idx = torch.where(target == 0)

    for t in thresholds:
        pred_labels = (preds >= t).float()
        correct = pred_labels == target

        # POSITIVES
        tp = correct[pos_label_idx].sum()
        n_p = (target == 1).sum()
        fn = n_p - tp
        # NEGATIVES
        tn = correct[neg_label_idx].sum()
        n_n = (target == 0).sum()
        fp = n_n - tn
        ###################################3
        # Balanced Accuracy
        ###################################3
        # TPR, TNR
        tpr = tp / n_p
        tnr = tn / n_n
        # BA
        ba_tmp = (tpr + tnr) / 2
        ba.append(ba_tmp)
        ###################################3
        # F1
        ###################################3
        precision1 = tp / (tp + fp + EPS)
        recall1 = tp / (tp + fn + EPS)
        f1_1 = 2 * precision1 * recall1 / (precision1 + recall1 + EPS)
        prec1.append(precision1)
        rec1.append(recall1)
        auc1.append(precision1 * recall1)

        precision0 = tn / (tn + fn + EPS)
        recall0 = tn / (tn + fp + EPS)
        f1_0 = 2 * precision0 * recall0 / (precision0 + recall0 + EPS)
        prec0.append(precision0)
        rec0.append(recall0)
        auc0.append(precision0 * recall0)

        f1w = (f1_0 * n_n + f1_1 * n_p) / (n_n + n_p)
        f1.append(f1w)

    return {
        "bacc": torch.stack(ba),
        "f1": torch.stack(f1),
        "prec1": torch.stack(prec1),
        "rec1": torch.stack(rec1),
        "prec0": torch.stack(prec0),
        "rec0": torch.stack(rec0),
        "auc0": torch.stack(auc0),
        "auc1": torch.stack(auc1),
        "thresholds": thresholds,
    }


def find_threshold(
    model: VAPModel,
    dloader: DataLoader,
    savepath: str,
    min_thresh: float = 0.01,
    cfg_dict: dict = None,
):
    """Find the best threshold using PR-curves"""

    def get_best_thresh(curves, metric, measure, min_thresh):
        ts = curves[metric]["thresholds"]
        over = min_thresh <= ts
        under = ts <= (1 - min_thresh)
        w = torch.where(torch.logical_and(over, under))
        values = curves[metric][measure][w]
        ts = ts[w]
        _, best_idx = values.max(0)
        return ts[best_idx]

    print("#" * 60)
    print("Finding Thresholds (val-set)...")
    print("#" * 60)

    # Init metric:
    model.test_metric = model.init_metric(
        bc_pred_pr_curve=True,
        shift_pred_pr_curve=True,
        long_short_pr_curve=True,
    )

    # Find Thresholds
    _trainer = pl.Trainer(
        deterministic=True,
        callbacks=[SymmetricSpeakersCallback()],
        **cfg_dict
    )

    _ = _trainer.test(model, dataloaders=dloader)

    ############################################
    predictions = {}
    if hasattr(model.test_metric, "long_short_pr"):
        predictions["long_short"] = {
            "preds": torch.cat(model.test_metric.long_short_pr.preds),
            "target": torch.cat(model.test_metric.long_short_pr.target),
        }
    if hasattr(model.test_metric, "bc_pred_pr"):
        predictions["bc_preds"] = {
            "preds": torch.cat(model.test_metric.bc_pred_pr.preds),
            "target": torch.cat(model.test_metric.bc_pred_pr.target),
        }
    if hasattr(model.test_metric, "shift_pred_pr"):
        predictions["shift_preds"] = {
            "preds": torch.cat(model.test_metric.shift_pred_pr.preds),
            "target": torch.cat(model.test_metric.shift_pred_pr.target),
        }

    ############################################
    # Curves
    curves = {}
    for metric in ["bc_preds", "long_short", "shift_preds"]:
        curves[metric] = get_curves(
            preds=predictions[metric]["preds"], target=predictions[metric]["target"]
        )

    ############################################
    # find best thresh
    bc_pred_threshold = None
    shift_pred_threshold = None
    long_short_threshold = None
    if "bc_preds" in curves:
        bc_pred_threshold = get_best_thresh(curves, "bc_preds", "f1", min_thresh)
    if "shift_preds" in curves:
        shift_pred_threshold = get_best_thresh(curves, "shift_preds", "f1", min_thresh)
    if "long_short" in curves:
        long_short_threshold = get_best_thresh(curves, "long_short", "f1", min_thresh)

    thresholds = {
        "pred_shift": shift_pred_threshold,
        "pred_bc": bc_pred_threshold,
        "short_long": long_short_threshold,
    }

    th = {k: v.item() for k, v in thresholds.items()}
    # torch.save(prediction, join(savepath, "predictions.pt"))
    write_json(th, join(savepath, "thresholds.json"))
    torch.save(curves, join(savepath, "curves.pt"))
    print("Saved Thresholds -> ", join(savepath, "thresholds.json"))
    print("Saved Curves -> ", join(savepath, "curves.pt"))
    return thresholds


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
        print('Press Enter to continue (in 10 seconds): ')
        i, o, e = select.select( [sys.stdin], [], [], 10 )
    else:
        checkpoint = args.checkpoint

    cfg_dict = configs["cfg_dict"]
    savepath = get_savepath(args, configs)

    if configs["args"].devices is None:
        gpu_devices = -1
    else:
        gpu_devices = [int(d.strip()) for d in configs["args"].devices.split(",")]
    
    if configs["args"].auto_select_gpus is not None and configs["args"].auto_select_gpus == 1:
        auto_select_gpus = True
        gpu_devices = int(configs["args"].devices)
    else:
        auto_select_gpus = False
    
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
        train_path=dconf.train_path,
        val_path=dconf.val_path,
        test_path=dconf.test_path,
        horizon=2,
        batch_size=dconf.batch_size,
        num_workers=dconf.num_workers,
    )
    dm.prepare_data()
    dm.setup("test")

    # # TODO: Do we still want to use zero-shot + threshold?
    # #########################################################
    # # Threshold
    # #########################################################
    # # Find the best thresholds (S-pred, BC-pred, S/L) on the validation set
    # thresholds = find_threshold(
    #     model, dm.val_dataloader(), savepath=savepath, min_thresh=MIN_THRESH
    # )
    # print(thresholds)
    # # #threshold_path = cfg.get("thresholds", None)
    # # if threshold_path is None:
    # #     thresholds = find_threshold(
    # #         model, dm.val_dataloader(), savepath=savepath, min_thresh=MIN_THRESH
    # #     )
    # # else:
    # #     print("Loading thresholds: ", threshold_path)
    # #     thresholds = read_json(threshold_path)

    #########################################################
    # Score
    #########################################################
    for pop in ["checkpoint", "seed", "gpus"]:
        cfg_dict.pop(pop)
    
    print(gpu_devices)
    if -1 not in gpu_devices:
        cfg_dict["accelerator"] = "gpu"
        cfg_dict["devices"] = gpu_devices
        cfg_dict["auto_select_gpus"] = auto_select_gpus
    else:
        cfg_dict["accelerator"] = "cpu"
        cfg_dict["devices"] = 1
    
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
    if cfg_dict["precision"] == 16:
        name += "_fp16"
    if cfg_dict["limit_test_batches"] is not None:
        nn = cfg_dict["limit_test_batches"] * dm.batch_size
        name += f"_nb-{nn}"

    filepath = join(savepath, name + ".csv")
    df.to_csv(filepath, index=False)
    print("Saved to -> ", filepath)

    # 結果を１つのファイルに追記する
    filepath = join(ROOT, "result-all.csv")
    with open(filepath, "a") as f:
        
        # 時間を書き込む
        f.write(datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
        f.write("\n")

        # モデルのファイルパスを書き込む
        f.write(checkpoint)
        f.write("\n")

        # テストデータの一覧ファイルを書き込む
        f.write(dconf.test_path)
        f.write("\n")

        if configs["data"].test_perturbation > 0:
            f.write("perturbation: " + str(configs["data"].test_perturbation))
            f.write("\n")
        
        f.write('Reference hs: 0=%d, 1=%d\n' % (model.test_hs_reference[0], model.test_hs_reference[1]))
        f.write('Confusion matrix hs:\n')
        for k, v in model.test_hs_confusion_matrix.items():
            f.write("%s -> %d\n" % (k, v))
        
        f.write('Reference hs2: 0=%d, 1=%d\n' % (model.test_hs2_reference[0], model.test_hs2_reference[1]))
        f.write('Confusion matrix hs2:\n')
        for k, v in model.test_hs2_confusion_matrix.items():
            f.write("%s -> %d\n" % (k, v))
        
        f.write('Reference shift2: 0=%d, 1=%d\n' % (model.test_pred_shift2_reference[0], model.test_pred_shift2_reference[1]))
        f.write('Confusion matrix shift2:\n')
        for k, v in model.test_pred_shift2_confusion_matrix.items():
            f.write("%s -> %d\n" % (k, v))

        f.write('Reference bc2: 0=%d, 1=%d\n' % (model.test_pred_backchannel2_reference[0], model.test_pred_backchannel2_reference[1]))
        f.write('Confusion matrix bc2:\n')
        for k, v in model.test_pred_backchannel2_confusion_matrix.items():
            f.write("%s -> %d\n" % (k, v))
        
        ave_inf_time = np.mean(model.test_inference_time_all)
        f.write('Inference Time by: %s\n' % cfg_dict["accelerator"])
        f.write('Average Inference Time [msec]: %f\n' % ave_inf_time)
        f.write('Average Inference Time (Hz): %f\n' % (1. / (ave_inf_time/1000.)))

        # dafをcsv形式で書き込む
        df.to_csv(f, header=True, index=False, sep=",")
        f.write("\n")
    
    print("Added to -> ", filepath)
    

if __name__ == "__main__":
    evaluate()
