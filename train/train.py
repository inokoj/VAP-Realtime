from argparse import ArgumentParser
from os import environ
from os import path
import os
from typing import Dict
from dataclasses import dataclass
import time

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from torchmetrics.classification import Accuracy, F1Score

from datamodule import VapDataModule
from callbacks import SymmetricSpeakersCallback, AudioAugmentationCallback
from events import TurnTakingEvents, EventConfig
from model import VapGPT, VapConfig

torch.set_float32_matmul_precision("medium")

@dataclass
class OptConfig:
    learning_rate: float = 3.63e-4
    find_learning_rate: bool = False
    betas = [0.9, 0.999]
    weight_decay: float = 0.001
    lr_scheduler_interval: str = "step"
    lr_scheduler_freq: int = 100
    lr_scheduler_tmax: int = 2500
    lr_scheduler_patience: int = 2
    lr_scheduler_factor: float = 0.5

    # early stopping
    early_stopping: int = 1
    patience: int = 10
    
    monitor: str = "val_loss"
    mode: str = "min"
    save_top_k: int = -1
    max_epochs: int = -1

    saved_dir: str = "/n/work1/inoue/vap/runs"

    @staticmethod
    def add_argparse_args(parser):
        for k, v in OptConfig.__dataclass_fields__.items():
            parser.add_argument(f"--opt_{k}", type=v.type, default=v.default)
        return parser

    @staticmethod
    def args_to_conf(args):
        return OptConfig(
            **{
                k.replace("opt_", ""): v
                for k, v in vars(args).items()
                if k.startswith("opt_")
            }
        )


@dataclass
class DataConfig:
    train_path: str = "../vap_dataset/data/sliding_train.csv"
    val_path: str = "../vap_dataset/data/sliding_val.csv"
    test_path: str = "../vap_dataset/data/sliding_test.csv"
    flip_channels: bool = True
    flip_probability: float = 0.5
    mask_vad: bool = True
    mask_vad_probability: float = 0.5
    #batch_size: int = 16
    batch_size: int = 8
    num_workers: int = 2

    # not used for datamodule
    audio_duration: float = 20

    #perturbation: str = "none"

    test_perturbation: int = 0
    # 0: no perturbation
    # 1: flat pitch
    # 2: low pass

    @staticmethod
    def add_argparse_args(parser):
        for k, v in DataConfig.__dataclass_fields__.items():
            parser.add_argument(f"--data_{k}", type=v.type, default=v.default)
        return parser

    @staticmethod
    def args_to_conf(args):
        return DataConfig(
            **{
                k.replace("data_", ""): v
                for k, v in vars(args).items()
                if k.startswith("data_")
            }
        )


def get_args():
    parser = ArgumentParser("VoiceActivityProjection")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    #parser = pl.Trainer.add_argparse_args(parser)
    parser = OptConfig.add_argparse_args(parser)
    parser = DataConfig.add_argparse_args(parser)
    parser, fields_added = VapConfig.add_argparse_args(parser)
    parser, fields_added = EventConfig.add_argparse_args(parser, fields_added)
    parser.add_argument("--devices", type=str, default='0')
    args = parser.parse_args()

    model_conf = VapConfig.args_to_conf(args)
    opt_conf = OptConfig.args_to_conf(args)
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
        "opt": opt_conf,
        "data": data_conf,
    }


def get_run_name(configs) -> str:
    s = "VapGPT"
    s += f"_{configs['model'].frame_hz}Hz"
    s += f"_ad{configs['data'].audio_duration}s"
    s += f"_{configs['model'].channel_layers}"
    s += str(configs["model"].cross_layers)
    s += str(configs["model"].num_heads)
    
    # if configs["model"].encoder_type == "cpc":
    #     s += "_cpc"
        
    #     # For original CPC model
    #     if configs["model"].cpc_model_pt != "" and configs["model"].cpc_model_pt != "default":
    #         #print(os.path.basename(configs["model"].cpc_model_pt))
    #         epoch_ = os.path.basename(configs["model"].cpc_model_pt).split('_')[1].split('.')[0]
    #         data_ = configs["model"].cpc_model_pt.split('/')[-2]
    #         s += '_' + data_ + '_' + epoch_
    
    if configs["model"].freeze_encoder == 0:
        s += "_enc-tuned"
    
    return s


def train() -> None:
    configs = get_args()
    cfg_dict = configs["cfg_dict"]

    pl.seed_everything(cfg_dict["seed"])
    local_rank = environ.get("LOCAL_RANK", 0)

    model = VAPModel(
        configs["model"], opt_conf=configs["opt"], event_conf=configs["event"]
    )

    name = get_run_name(configs)

    dconf = configs["data"]
    dm = VapDataModule(
        train_path=dconf.train_path,
        val_path=dconf.val_path,
        test_path=None,
        horizon=2,
        batch_size=dconf.batch_size,
        num_workers=dconf.num_workers,
        frame_hz=configs["model"].frame_hz,
    )
    dm.prepare_data()

    if configs["args"].devices is None:
        gpu_devices = -1
    else:
        gpu_devices = [int(d.strip()) for d in configs["args"].devices.split(",")]
    
    # if configs["args"].auto_select_gpus is not None and configs["args"].auto_select_gpus == 1:
    #     auto_select_gpus = True
    #     gpu_devices = int(configs["args"].devices)
    # else:
    #     auto_select_gpus = False
    
    # if cfg_dict["debug"]:
    #     environ["WANDB_MODE"] = "offline"
    #     print("DEBUG -> OFFLINE MODE")

    # if cfg_dict["fast_dev_run"]:
    #     print("NAME: " + name)
    #     for n in ["logger", "strategy", "debug", "seed", "wandb_project"]:
    #         cfg_dict.pop(n)
    #     trainer = pl.Trainer(**cfg_dict)
    #     trainer.fit(model, datamodule=dm)
    # else:

    oconf = configs["opt"]

    # Callbacks & Logger
    logger = None

    # Crate save dir
    if path.exists(oconf.saved_dir) == False:
        os.makedirs(oconf.saved_dir)
    else:
        list_filenames = []
        for f in os.listdir(oconf.saved_dir):
            if name + '-epoch' in f:
                tmp = os.path.join(oconf.saved_dir, f)
                print(tmp)
                list_filenames.append(tmp)
        
        if len(list_filenames) > 1:
            
            # Delete esiisting files
            prompt = 'Already existing files in the save directory. Do you want to delete them? [yes/no]'
            while True:
                user_input = input(prompt).lower()
                if user_input == 'yes' or user_input == 'no':
                    break
                else:
                    print('Please enter either "yes" or "no".')
            if user_input == 'yes':
                for f in list_filenames:
                    os.remove(f)
                    print("Removed file: " + f)

    callbacks = [
        ModelCheckpoint(
            dirpath=oconf.saved_dir,
            mode=oconf.mode,
            monitor=oconf.monitor,
            save_top_k=oconf.save_top_k,
            auto_insert_metric_name=False,
            filename=name + "-epoch{epoch}-val_{val_loss:.5f}",
        ),
        SymmetricSpeakersCallback()
    ]

    if configs["model"].context_limit_cpc_sec < 0:
        callbacks.append(AudioAugmentationCallback(device="cpu"))
    
    if oconf.early_stopping == 1:
        print("Eearly Stopping applied")
        callbacks.append(
            EarlyStopping(
                monitor=oconf.monitor,
                mode=oconf.mode,
                patience=oconf.patience,
                strict=True,  # crash if "monitor" is not found in val metrics
                verbose=False,
            )
        )

    # if not cfg_dict["debug"]:
    #     logger = WandbLogger(
    #         project=cfg_dict["wandb_project"],
    #         name=name,
    #         log_model=False,
    #         save_dir="runs",
    #     )
    #     callbacks.append(LearningRateMonitor())

    if local_rank == 0:
        print("#" * 40)
        print(f"Early stopping (patience={oconf.patience})")
        print("#" * 40)

    # Find Best Learning Rate
    if oconf.find_learning_rate:
        trainer = pl.Trainer(accelerator="gpu", devices=gpu_devices)
        lr_finder = trainer.tuner.lr_find(model, dm)
        model.learning_rate = lr_finder.suggestion()
    print("Learning Rate: ", model.opt_conf.learning_rate)
    print("#" * 40)

    # Actual Training
    if torch.cuda.is_available():
        cfg_dict["accelerator"] = "gpu"
        cfg_dict["devices"] = gpu_devices
    else:
        cfg_dict["accelerator"] = "cpu"
        cfg_dict["devices"] = 1

    for n in ["logger", "strategy", "debug", "seed", "wandb_project"]:
        if n in cfg_dict:
            cfg_dict.pop(n)
    
    cfg_dict['max_epochs'] = oconf.max_epochs
    
    if configs["model"].context_limit_cpc_sec > 0:
        bool_find_unused_parameters = True
    else:
        bool_find_unused_parameters = False
    
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        strategy=DDPStrategy(find_unused_parameters=bool_find_unused_parameters),
        **cfg_dict
    )

    #ckpt_path = None

    # if path.exists(fileame_checkpoint):
    #     print("Loading checkpoint: ", fileame_checkpoint)
    #     ckpt_path = fileame_checkpoint
    # else:
    #     ckpt_path = None

    trainer.fit(model, datamodule=dm)


# Used in training (LightningModule) but not required for inference
class VAPModel(VapGPT, pl.LightningModule):
    def __init__(self, conf, opt_conf=None, event_conf=None):
        super().__init__(conf)

        self.opt_conf = opt_conf
        self.event_conf = event_conf

        # Training params
        self.save_hyperparameters()

        # Metrics
        self.event_extractor = None
        if event_conf is not None:
            # self.zero_shot = ZeroShot(bin_times=conf.bin_times, frame_hz=conf.frame_hz)
            self.event_extractor = TurnTakingEvents(event_conf)
        
        self.training_step_outputs = []
        # self.test_perturbation = 0

        # self.test_hs_reference = [0, 0]
        # self.test_hs_confusion_matrix = {"ref=0:pre=0": 0, "ref=0:pre=1": 0, "ref=1:pre=0": 0, "ref=1:pre=1": 0}

        # self.test_hs2_reference = [0, 0]
        # self.test_hs2_confusion_matrix = {"ref=0:pre=0": 0, "ref=0:pre=1": 0, "ref=1:pre=0": 0, "ref=1:pre=1": 0}

        # self.test_pred_shift2_reference = [0, 0]
        # self.test_pred_shift2_confusion_matrix = {"ref=0:pre=0": 0, "ref=0:pre=1": 0, "ref=1:pre=0": 0, "ref=1:pre=1": 0}

        # self.test_pred_backchannel2_reference = [0, 0]
        # self.test_pred_backchannel2_confusion_matrix = {"ref=0:pre=0": 0, "ref=0:pre=1": 0, "ref=1:pre=0": 0, "ref=1:pre=1": 0}

        # self.test_inference_time_all = []

    def get_metrics(self):
        metrics = {"acc": {}, "f1": {}}

        ACC_TASK = 'multiclass'
        ACC_AVERAGE = 'none'

        metrics["acc"]["hs"] = Accuracy(
            task=ACC_TASK, num_classes=2, average=ACC_AVERAGE
        ).to(self.device)
        
        metrics["acc"]["hs2"] = Accuracy(
            task=ACC_TASK, num_classes=2, average=ACC_AVERAGE
        ).to(self.device)
        
        metrics["acc"]["ls"] = Accuracy(
            task=ACC_TASK, num_classes=2, average=ACC_AVERAGE
        ).to(self.device)
        
        metrics["acc"]["sp"] = Accuracy(
            task=ACC_TASK, num_classes=2, average=ACC_AVERAGE
        ).to(self.device)
        
        metrics["acc"]["bp"] = Accuracy(
            task=ACC_TASK, num_classes=2, average=ACC_AVERAGE
        ).to(self.device)

        metrics["acc"]["sp2"] = Accuracy(
            task=ACC_TASK, num_classes=2, average=ACC_AVERAGE
        ).to(self.device)
        
        metrics["acc"]["bp2"] = Accuracy(
            task=ACC_TASK, num_classes=2, average=ACC_AVERAGE
        ).to(self.device)

        metrics["f1"]["hs"] = F1Score(
            task="multiclass",
            num_classes=2,
            average="weighted",
        ).to(self.device)

        metrics["f1"]["hs2"] = F1Score(
            task="multiclass",
            num_classes=2,
            average="weighted",
        ).to(self.device)
        
        metrics["f1"]["ls"] = F1Score(
            task="multiclass",
            num_classes=2,
            average="weighted",
        ).to(self.device)
        
        metrics["f1"]["sp"] = F1Score(
            task="multiclass",
            num_classes=2,
            average="weighted",
        ).to(self.device)
        
        metrics["f1"]["bp"] = F1Score(
            task="multiclass",
            num_classes=2,
            average="weighted",
        ).to(self.device)

        metrics["f1"]["sp2"] = F1Score(
            task="multiclass",
            num_classes=2,
            average="weighted",
        ).to(self.device)
        
        metrics["f1"]["bp2"] = F1Score(
            task="multiclass",
            num_classes=2,
            average="weighted",
        ).to(self.device)

        metrics["f1"]["lid"] = F1Score(
            task="multiclass",
            num_classes=self.conf.lid_classify_num_class,
            average="weighted",
        ).to(self.device)

        return metrics

    def metrics_step(self, preds, targets, split="val"):
        
        m = self.val_metrics if split == "val" else self.test_metrics

        # The metrics don't work if the predictions are not rounded
        # I don't know why...
        if preds["hs"] is not None:
            if len(targets["hs"]) >= 1:
                m["f1"]["hs"].update(preds=preds["hs"].round(), target=targets["hs"])
                m["acc"]["hs"].update(preds=preds["hs"].round(), target=targets["hs"])
        
        if "hs2" in preds:
            if preds["hs2"] is not None:
                if len(targets["hs2"]) >= 1:
                    m["f1"]["hs2"].update(preds=preds["hs2"].round(), target=targets["hs2"])
                    m["acc"]["hs2"].update(preds=preds["hs2"].round(), target=targets["hs2"])

        if preds["ls"] is not None:
            if len(targets["ls"]) >= 1:
                m["f1"]["ls"].update(preds=preds["ls"].round(), target=targets["ls"])
                m["acc"]["ls"].update(preds=preds["ls"].round(), target=targets["ls"])

        if preds["pred_shift"] is not None:
            if len(targets["pred_shift"]) >= 1:
                m["f1"]["sp"].update(preds=preds["pred_shift"].round(), target=targets["pred_shift"])
                m["acc"]["sp"].update(preds=preds["pred_shift"].round(), target=targets["pred_shift"])

        if preds["pred_backchannel"] is not None:
            if len(targets["pred_backchannel"]) >= 1:
                m["f1"]["bp"].update(preds=preds["pred_backchannel"].round(), target=targets["pred_backchannel"])
                m["acc"]["bp"].update(preds=preds["pred_backchannel"].round(), target=targets["pred_backchannel"])
        
        if "pred_shift2" in preds:
            if preds["pred_shift2"] is not None:
                if len(targets["pred_shift2"]) >= 1:
                    m["f1"]["sp2"].update(preds=preds["pred_shift2"].round(), target=targets["pred_shift2"])
                    m["acc"]["sp2"].update(preds=preds["pred_shift2"].round(), target=targets["pred_shift2"])

        if "pred_backchannel2" in preds:
            if preds["pred_backchannel2"] is not None:
                if len(targets["pred_backchannel2"]) >= 1:
                    m["f1"]["bp2"].update(preds=preds["pred_backchannel2"].round(), target=targets["pred_backchannel2"])
                    m["acc"]["bp2"].update(preds=preds["pred_backchannel2"].round(), target=targets["pred_backchannel2"])
        
        if preds["lid"] is not None:
            if len(targets["lid"]) >= 1:
                m["f1"]["lid"].update(preds=preds["lid"], target=targets["lid"])

    def metrics_epoch(self, split="val"):
        if split == "val":
            m = self.val_metrics
        else:
            m = self.test_metrics

        # f1 = {}
        # for name, metric in m["f1"].items():
        #     f1[name] = metric.compute()

        #     #support = metric._num_examples[name]

        #     metric.reset()

        # # Accuracy
        # acc = {}
        # for name, metric in m["acc"].items():
        #     a, b = metric.compute()
        #     acc[name] = [a, b]
        #     metric.reset()

        # self.log(f"{split}_hs", {"acc": acc["hs"][1], "f1w": f1["hs"]}, prog_bar=True, sync_dist=True)
        # self.log(f"{split}_hs2", {"acc": acc["hs2"][1], "f1w": f1["hs2"]}, prog_bar=True, sync_dist=True)
        
        # self.log(f"{split}_pred_sh", {"acc": acc["sp"][1], "f1w": f1["sp"]}, sync_dist=True)
        # self.log(f"{split}_pred_sh2", {"acc": acc["sp2"][1], "f1w": f1["sp2"]}, sync_dist=True)
        
        # self.log(f"{split}_pred_bc", {"acc": acc["bp"][1], "f1w": f1["bp"]}, sync_dist=True)
        # self.log(f"{split}_pred_bc2", {"acc": acc["bp2"][1], "f1w": f1["bp2"]}, sync_dist=True)

        # self.log(f"{split}_ls", {"short": acc["ls"][1]}, sync_dist=True)
        
        # self.log(f"{split}_lid", {"f1w": f1["lid"]}, sync_dist=True)

        # if split == "test":
        #     # hs2, sh2, bc2のBalanced accuracyを計算
        #     # Balanced accuracy = (TPR + TNR) / 2

        #     hs2_tp = self.test_hs2_confusion_matrix['ref=1:pre=1']
        #     hs2_fp = self.test_hs2_confusion_matrix['ref=0:pre=1']
        #     hs2_fn = self.test_hs2_confusion_matrix['ref=1:pre=0']
        #     hs2_tn = self.test_hs2_confusion_matrix['ref=0:pre=0']
        #     hs2_balanced_accuracy = (hs2_tp / (hs2_tp + hs2_fn) + hs2_tn / (hs2_tn + hs2_fp)) / 2

        #     pred_sh2_tp = self.test_pred_shift2_confusion_matrix['ref=1:pre=1']
        #     pred_sh2_fp = self.test_pred_shift2_confusion_matrix['ref=0:pre=1']
        #     pred_sh2_fn = self.test_pred_shift2_confusion_matrix['ref=1:pre=0']
        #     pred_sh2_tn = self.test_pred_shift2_confusion_matrix['ref=0:pre=0']
        #     pred_sh2_balanced_accuracy = (pred_sh2_tp / (pred_sh2_tp + pred_sh2_fn) + pred_sh2_tn / (pred_sh2_tn + pred_sh2_fp)) / 2

        #     pred_bc2_tp = self.test_pred_backchannel2_confusion_matrix['ref=1:pre=1']
        #     pred_bc2_fp = self.test_pred_backchannel2_confusion_matrix['ref=0:pre=1']
        #     pred_bc2_fn = self.test_pred_backchannel2_confusion_matrix['ref=1:pre=0']
        #     pred_bc2_tn = self.test_pred_backchannel2_confusion_matrix['ref=0:pre=0']
        #     pred_bc2_balanced_accuracy = (pred_bc2_tp / (pred_bc2_tp + pred_bc2_fn) + pred_bc2_tn / (pred_bc2_tn + pred_bc2_fp)) / 2

        #     self.log(f"{split}_hs2_balanced_accuracy", hs2_balanced_accuracy, sync_dist=True)
        #     self.log(f"{split}_pred_sh2_balanced_accuracy", pred_sh2_balanced_accuracy, sync_dist=True)
        #     self.log(f"{split}_pred_bc2_balanced_accuracy", pred_bc2_balanced_accuracy, sync_dist=True)

    def shared_step(
        self, batch: Dict, reduction: str = "mean"
    ) -> Dict[str, torch.Tensor]:
        """
        Arguments:
            batch:      dict, containing 'waveform', va, va_history

        Returns:
            out:        dict, ['logits', 'vad', 'vap_loss', 'vad_loss']
        """
        
        labels = self.objective.get_labels(batch["vad"])
        out = self(waveform=batch["waveform"])

        out["vap_loss"] = self.objective.loss_vap(out["logits"], labels, reduction=reduction)
        out["vad_loss"] = self.objective.loss_vad(out["vad"], batch["vad"])

        return out

    def configure_optimizers(self) -> Dict:
        assert self.opt_conf is not None, "configure_optimizers: No Opt conf!"
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.opt_conf.learning_rate,
            betas=self.opt_conf.betas,
            weight_decay=self.opt_conf.weight_decay,
        )
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode="min",
                factor=self.opt_conf.lr_scheduler_factor,
                patience=self.opt_conf.lr_scheduler_patience,
            ),
            "monitor": "val_loss",
        }
        return {"optimizer": opt, "lr_scheduler": lr_scheduler}

    def training_step(self, batch, batch_idx, **kwargs):
        
        out = self.shared_step(batch)
        batch_size = batch["waveform"].shape[0]

        self.log("loss_train", out["vap_loss"], batch_size=batch_size, on_epoch=True, on_step=True, sync_dist=True)
        self.log("loss_train_va", out["vad_loss"], batch_size=batch_size, on_epoch=True, on_step=True, sync_dist=True)        
        loss = out["vap_loss"] + out["vad_loss"]

        return {"loss": loss}

    def validation_step(self, batch, batch_idx, **kwargs):
        """validation step"""
        if not hasattr(self, "val_metrics"):
            self.val_metrics = self.get_metrics()

        out = self.shared_step(batch)
        batch_size = batch["waveform"].shape[0]
        
        self.log("val_loss", out["vap_loss"], batch_size=batch_size, on_epoch=True, on_step=True, sync_dist=True)
        self.log("val_loss_va", out["vad_loss"], batch_size=batch_size, on_epoch=True, on_step=True, sync_dist=True)

        # self.log("val_loss_step", out["vap_loss"], on_step=True, sync_dist=True)
        # self.log("val_loss_va_step", out["vad_loss"], on_step=True, sync_dist=True)

        # Event Metrics
        if self.event_extractor is not None:
            events = self.event_extractor(batch["vad"])
            # probs = self.zero_shot.get_probs(out["logits"], batch["vad"])
            # preds, targets = self.zero_shot.extract_prediction_and_targets(
            #     p=probs["p"], p_bc=probs["p_bc"], events=events
            # )    
            probs = self.objective.get_probs(out["logits"])
            preds, targets = self.objective.extract_prediction_and_targets(
                p_now=probs["p_now"], p_fut=probs["p_future"], events=events
            )
            # print(probs)
            # print(preds)
            # print(targets)
            # input('Check1')
            self.metrics_step(preds, targets, split="val")

    def on_validation_epoch_end(self, *_):
        if hasattr(self, "val_metrics"):
            self.metrics_epoch("val")

    def test_step(self, batch, batch_idx, **kwargs):

        """validation step"""
        if not hasattr(self, "test_metrics"):
            self.test_metrics = self.get_metrics()

            for name, events in self.test_metrics.items():
                for event, metric in events.items():
                    strname = f"test_{name}_{event}"
                    self.register_module(strname, metric)
        
        # if not hasattr(self, "test_sh_metrics_conf"):
        #     self.test_sh_metrics_conf = MulticlassConfusionMatrix(2)

        #
        # Perterbation
        #
        
        # import torchaudio
        # torchaudio.save("flatintensity_sample-before.wav", batch["waveform"][0, :, :].cpu(), 16000)

        # #pert = FlatIntensity(min_intensity=20)
        # if self.test_perturbation == 1:
        #     pert = FlatPitch()
        #     batch["waveform"] = pert(batch["waveform"])
        
        # if self.test_perturbation == 2:

        #     # import torchaudio
        #     # torchaudio.save("lowpass_sample-before.wav", batch["waveform"][0, :, :].cpu(), 16000)

        #     pert = LowPass()
        #     batch["waveform"] = pert(batch["waveform"])

        #     # # import torchaudio
        #     # torchaudio.save("lowpass_sample-after.wav", batch["waveform"][0, :, :].cpu(), 16000)
        
        # pert = FlatPitch()
        # batch["waveform"] = pert(batch["waveform"])
        
        # # torchaudio.save("flatintensity_sample-after.wav", batch["waveform"][0, :, :].cpu(), 16000)
        # # input()

        out = self.shared_step(batch)
        # self.test_inference_time_all.append(out["inf_time"])
        batch_size = batch["waveform"].shape[0]
        
        self.log("test_loss", out["vap_loss"], batch_size=batch_size, sync_dist=True)
        self.log("test_loss_va", out["vad_loss"], batch_size=batch_size, sync_dist=True)

        # Event Metrics
        if self.event_extractor is not None:
            
            events = self.event_extractor(batch["vad"])
            
            # probs = self.zero_shot.get_probs(out["logits"], batch["vad"])
            # preds, targets = self.zero_shot.extract_prediction_and_targets(
            #     p=probs["p"], p_bc=probs["p_bc"], events=events
            # )
            
            probs_ojective = self.objective.get_probs(out["logits"])
            # # print(probs["p_now"])
            # # print(events)
            # # input()
            preds_objective, targets_objective = self.objective.extract_prediction_and_targets(
                p_now=probs_ojective["p_now"], p_fut=probs_ojective["p_future"], events=events
            )
            preds = preds_objective
            targets = targets_objective

            #preds["lid"] = preds_objective["lid"]
            #targets["lid"] = targets_objective["lid"]

            # print(batch['session'])
            # print(batch['vad'][0][:, 0].tolist())
            # print(batch['vad'][0][:, 1].tolist())
            # print(len(batch['vad'][0][:, 0].tolist()))
            # print(events)
            # print(events['shift'])
            # print(events['hold'])
            # print(preds['hs2'])
            # print(targets['hs2'])
            # input()

            # if 'hs2' in targets:
            #     if targets['hs2'] is not None:
            #         for r in targets['hs2']:
            #             self.test_hs2_reference[r] += 1
                    
            #         for p, t in zip(preds["hs2"], targets["hs2"]):
            #             if torch.isnan(p) or torch.isnan(t):
            #                 continue
            #             p_ = torch.round(p)
            #             label = "ref=%d:pre=%d" % (t, p_)
            #             self.test_hs2_confusion_matrix[label] += 1
            
            # if 'hs' in targets:
            #     if targets['hs'] is not None:
            #         for r in targets['hs']:
            #             self.test_hs_reference[r] += 1
                    
            #         for p, t in zip(preds["hs"], targets["hs"]):
            #             p_ = torch.round(p)
            #             label = "ref=%d:pre=%d" % (t, p_)
            #             self.test_hs_confusion_matrix[label] += 1
            
            # label_shift = "pred_shift2"
            # if label_shift in targets:
            #     if targets[label_shift] is not None:
            #         for r in targets[label_shift]:
            #             self.test_pred_shift2_reference[r] += 1
                    
            #         for p, t in zip(preds[label_shift], targets[label_shift]):
            #             p_ = torch.round(p)
            #             label = "ref=%d:pre=%d" % (t, p_)
            #             self.test_pred_shift2_confusion_matrix[label] += 1
            
            # label_backchannel = 'pred_backchannel2'
            # if label_backchannel in targets:
            #     if targets[label_backchannel] is not None:
            #         for r in targets[label_backchannel]:
            #             self.test_pred_backchannel2_reference[r] += 1
                    
            #         for p, t in zip(preds[label_backchannel], targets[label_backchannel]):
            #             p_ = torch.round(p)
            #             label = "ref=%d:pre=%d" % (t, p_)
            #             self.test_pred_backchannel2_confusion_matrix[label] += 1

            # print(batch['vad'][0][820-1:838+1, 0].tolist())
            # print(batch['vad'][0][820-1:838+1, 1].tolist())
            # print(preds['hs'])
            # print(targets['hs'])
            # input()
            # #print(batch["vad"])
            # # print(events)
            # # input()

            self.metrics_step(preds, targets, split="test")

    def on_validation_epoch_end(self, *_):
        if hasattr(self, "test_metrics"):
            self.metrics_epoch("test")


if __name__ == "__main__":
    train()
