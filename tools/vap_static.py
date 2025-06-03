import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional

from rvap.vap_main.encoder import EncoderCPC
from rvap.vap_main.modules import GPT, GPTStereo
from rvap.vap_main.objective import ObjectiveVAP

# from wav import WavLoadForVAP
import time

import numpy as np

import copy
import argparse

from os import environ
environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import rvap.common.util as util

torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(mode=True)

torch.manual_seed(0)

BIN_TIMES: list = [0.2, 0.4, 0.6, 0.8]

@dataclass
class VapConfig:
    sample_rate: int = 16000
    frame_hz: int = 50
    bin_times: List[float] = field(default_factory=lambda: BIN_TIMES)

    # Encoder (training flag)
    encoder_type: str = "cpc"
    wav2vec_type: str = "mms"
    hubert_model: str = "hubert_jp"
    freeze_encoder: int = 1  # stupid but works (--vap_freeze_encoder 1)
    load_pretrained: int = 1  # stupid but works (--vap_load_pretrained 1)
    only_feature_extraction: int = 0

    # GPT
    dim: int = 256
    channel_layers: int = 1
    cross_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1
    context_limit: int = -1

    context_limit_cpc_sec: float = -1

    # Added Multi-task
    lid_classify: int = 0   # 1...last layer, 2...middle layer
    lid_classify_num_class: int = 3
    lid_classify_adversarial: int = 0
    lang_cond: int = 0

    @staticmethod
    def add_argparse_args(parser, fields_added=[]):
        for k, v in VapConfig.__dataclass_fields__.items():
            if k == "bin_times":
                parser.add_argument(
                    f"--vap_{k}", nargs="+", type=float, default=v.default_factory()
                )
            else:
                parser.add_argument(f"--vap_{k}", type=v.type, default=v.default)
            fields_added.append(k)
        return parser, fields_added

    @staticmethod
    def args_to_conf(args):
        return VapConfig(
            **{
                k.replace("vap_", ""): v
                for k, v in vars(args).items()
                if k.startswith("vap_")
            }
        )

class VapGPT(nn.Module):

    def __init__(self, conf: Optional[VapConfig] = None):
        super().__init__()
        if conf is None:
            conf = VapConfig()
        self.conf = conf
        self.sample_rate = conf.sample_rate
        self.frame_hz = conf.frame_hz

        self.temp_elapse_time = []

        # Single channel
        self.ar_channel = GPT(
            dim=conf.dim,
            dff_k=3,
            num_layers=conf.channel_layers,
            num_heads=conf.num_heads,
            dropout=conf.dropout,
            context_limit=conf.context_limit,
        )

        # Cross channel
        self.ar = GPTStereo(
            dim=conf.dim,
            dff_k=3,
            num_layers=conf.cross_layers,
            num_heads=conf.num_heads,
            dropout=conf.dropout,
            context_limit=conf.context_limit,
        )

        self.objective = ObjectiveVAP(bin_times=conf.bin_times, frame_hz=conf.frame_hz)

        # Outputs
        # Voice activity objective -> x1, x2 -> logits ->  BCE
        self.va_classifier = nn.Linear(conf.dim, 1)

        if self.conf.lid_classify == 1:
            self.lid_classifier = nn.Linear(conf.dim, conf.lid_classify_num_class)

        elif self.conf.lid_classify == 2:
            self.lid_classifier_middle = nn.Linear(conf.dim*2, conf.lid_classify_num_class)

        if self.conf.lang_cond == 1:
            self.lang_condition = nn.Linear(conf.lid_classify_num_class, conf.dim)

        self.vap_head = nn.Linear(conf.dim, self.objective.n_classes)

    def load_encoder(self, cpc_model):

        # Audio Encoder
        self.encoder1 = EncoderCPC(
            load_pretrained=True if self.conf.load_pretrained == 1 else False,
            freeze=self.conf.freeze_encoder,
            cpc_model=cpc_model
        )
        self.encoder1 = self.encoder1.eval()

        self.encoder2 = EncoderCPC(
            load_pretrained=True if self.conf.load_pretrained == 1 else False,
            freeze=self.conf.freeze_encoder,
            cpc_model=cpc_model
        )
        self.encoder2 = self.encoder2.eval()

        if self.conf.freeze_encoder == 1:
            print('freeze encoder')
            self.encoder1.freeze()
            self.encoder2.freeze()

    @property
    def horizon_time(self):
        return self.objective.horizon_time

    def encode_audio(self, audio1: torch.Tensor, audio2: torch.Tensor) -> Tuple[Tensor, Tensor]:

        x1 = self.encoder1(audio1)  # speaker 1
        x2 = self.encoder2(audio2)  # speaker 2

        return x1, x2

    def vad_loss(self, vad_output, vad):
        return F.binary_cross_entropy_with_logits(vad_output, vad)

class VAPRealTimeStatic(nn.Module):

    BINS_P_NOW = [0, 1]
    BINS_PFUTURE = [2, 3]

    CALC_PROCESS_TIME_INTERVAL = 100

    def __init__(self, vap_model: str, cpc_model: str, device: torch.device, frame_rate: int, context_len_sec: float):
        super(VAPRealTimeStatic, self).__init__()

        conf = VapConfig()
        self.vap_gpt = VapGPT(conf)

        self.device = device

        sd = torch.load(vap_model, map_location=self.device)
        self.vap_gpt.load_encoder(cpc_model=cpc_model)
        self.vap_gpt.load_state_dict(sd, strict=False)

        # The downsampling parameters are not loaded by "load_state_dict"
        self.vap_gpt.encoder1.downsample[1].weight = nn.Parameter(sd['encoder.downsample.1.weight'])
        self.vap_gpt.encoder1.downsample[1].bias = nn.Parameter(sd['encoder.downsample.1.bias'])
        self.vap_gpt.encoder1.downsample[2].ln.weight = nn.Parameter(sd['encoder.downsample.2.ln.weight'])
        self.vap_gpt.encoder1.downsample[2].ln.bias = nn.Parameter(sd['encoder.downsample.2.ln.bias'])

        self.vap_gpt.encoder2.downsample[1].weight = nn.Parameter(sd['encoder.downsample.1.weight'])
        self.vap_gpt.encoder2.downsample[1].bias = nn.Parameter(sd['encoder.downsample.1.bias'])
        self.vap_gpt.encoder2.downsample[2].ln.weight = nn.Parameter(sd['encoder.downsample.2.ln.weight'])
        self.vap_gpt.encoder2.downsample[2].ln.bias = nn.Parameter(sd['encoder.downsample.2.ln.bias'])

        self.vap_gpt.to(self.device)
        self.vap_gpt = self.vap_gpt.eval()

        self.audio_contenxt_lim_sec = context_len_sec
        self.frame_rate = frame_rate

        # Context length of the audio embeddings (depends on frame rate)
        self.audio_context_len = int(self.audio_contenxt_lim_sec * self.frame_rate)

        self.sampling_rate = 16000
        self.frame_contxt_padding = 320 # Independe from frame size

        # Frame size
        # 10Hz -> 320 + 1600 samples
        # 20Hz -> 320 + 800 samples
        # 50Hz -> 320 + 320 samples
        self.audio_frame_size = self.sampling_rate // self.frame_rate + self.frame_contxt_padding

        self.current_x1_audio = []
        self.current_x2_audio = []

        self.result_p_now = 0.
        self.result_p_future = 0.
        self.result_last_time = -1

        self.result_vad = [0., 0.]

        self.process_time_abs = -1

        self.e1_context = []
        self.e2_context = []

        self.list_process_time_context = []
        self.last_interval_time = time.time()

    def forward(self, x1_: torch.Tensor, x2_: torch.Tensor, e1_context: torch.Tensor, e2_context: torch.Tensor):
        # vap_state_dict_jp_20hz_2500msec.pt
        #   data_left_frame.shape[x1]: (1120), data_right_frame.shape[x2]: (1120)
        #   x1_.shape: torch.Size([1, 1, 1120]), x2_.shape: torch.Size([1, 1, 1120])
        #
        # e1_context/e2_context に与えるテンソルについて
        #   1回目のforward時の e1_context/e2_context は torch.zeros([1, 1, 256])
        #   2回目のforward時の e1_context/e2_context は torch.tensor([1, 2, 256])
        #       :
        #   99回目のforward時の e1_context/e2_context は torch.tensor([1, 99, 256])
        #   100回目のforward時の e1_context/e2_context は torch.tensor([1, 99, 256])
        with torch.no_grad():
            # e1.shape: torch.Size([1, 1, 256])
            # e2.shape: torch.Size([1, 1, 256])
            e1, e2 = self.vap_gpt.encode_audio(audio1=x1_, audio2=x2_)

            # x1/x2 の形状遷移
            #   1回目のforward時の e1_context/e2_context は torch.zeros([1, 2, 256])
            #   2回目のforward時の e1_context/e2_context は torch.tensor([1, 3, 256])
            #       :
            #   99回目のforward時の e1_context/e2_context は torch.tensor([1, 100, 256])
            #   100回目のforward時の e1_context/e2_context は torch.tensor([1, 100, 256])
            x1_ = torch.cat([e1_context, e1], dim=1).to(self.device)
            x2_ = torch.cat([e2_context, e2], dim=1).to(self.device)

            o1 = self.vap_gpt.ar_channel(x1_, attention=False)
            o2 = self.vap_gpt.ar_channel(x2_, attention=False)
            out = self.vap_gpt.ar(o1["x"], o2["x"], attention=False)

            # Outputs
            logits: torch.Tensor = self.vap_gpt.vap_head(out["x"])

            vad1: torch.Tensor = self.vap_gpt.va_classifier(o1["x"])
            vad2: torch.Tensor = self.vap_gpt.va_classifier(o2["x"])

            probs = logits.softmax(dim=-1)

            p_now = self.vap_gpt.objective.probs_next_speaker_aggregate(
                probs,
                from_bin=self.BINS_P_NOW[0],
                to_bin=self.BINS_P_NOW[-1]
            )

            p_future = self.vap_gpt.objective.probs_next_speaker_aggregate(
                probs,
                from_bin=self.BINS_PFUTURE[0],
                to_bin=self.BINS_PFUTURE[1]
            )

            vad1 = vad1.sigmoid()[::, -1]
            vad2 = vad2.sigmoid()[::, -1]

            # p_now.shape: torch.Size([1, 68, 2])
            # p_future.shape: torch.Size([1, 68, 2])
            # vad1.shape: torch.Size([1, 1])
            # vad2.shape: torch.Size([1, 1])
            # e1.shape: torch.Size([1, 1, 256])
            #   モデルをコールするメインプロセス側で 1〜99 の範囲で dim=1 を cat したテンソルを次の推論時に入力として渡す
            #   初期値は torch.zeros([1,1,256], dtype=troch.float32) で良い
            # e2.shape: torch.Size([1, 1, 256])
            #   モデルをコールするメインプロセス側で 1〜99 の範囲で dim=1 を cat したテンソルを次の推論時に入力として渡す
            #   初期値は torch.zeros([1,1,256], dtype=troch.float32) で良い
            #
            # e1_context/e2_context に与えるテンソルについて
            #   1回目のforward時の e1_context/e2_context は torch.zeros([1, 1, 256])
            #   2回目のforward時の e1_context/e2_context は torch.zeros([1, 2, 256])
            #       :
            #   99回目のforward時の e1_context/e2_context は torch.zeros([1, 99, 256])
            #   100回目のforward時の e1_context/e2_context は torch.zeros([1, 99, 256])
            return p_now[:, -1, :], p_future[:, -1, :], vad1, vad2, e1, e2