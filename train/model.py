import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional

from encoder import EncoderCPC
from objective import ObjectiveVAP
from modules import GPT, GPTStereo
from utils import (
    everything_deterministic,
    vad_fill_silences,
    vad_omit_spikes,
)

BIN_TIMES: list = [0.2, 0.4, 0.6, 0.8]

everything_deterministic()

# TODO: @dataclass and CLI arguments or is hydra the way to go?
# TODO: Easy finetune task
# TODO: What to evaluate


def load_older_state_dict(
    path="example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.ckpt",
):
    sd = torch.load(path)["state_dict"]
    new_sd = {}
    for k, v in sd.items():
        if "VAP.codebook" in k:
            continue
        if "vap_head" in k:
            k = k.replace("vap_head.projection_head", "vap_head")
        new_sd[k.replace("net.", "")] = v
    return new_sd


@dataclass
class VapConfig:
    sample_rate: int = 16_000
    frame_hz: int = 50
    bin_times: List[float] = field(default_factory=lambda: BIN_TIMES)

    # Encoder (training flag)
    encoder_type: str = "cpc"
    cpc_model_pt: str = "default"
    
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

        # Audio Encoder
        if self.conf.encoder_type == "cpc":
            self.encoder = EncoderCPC(
                cpc_model_pt=self.conf.cpc_model_pt,
                load_pretrained=True if conf.load_pretrained == 1 else False,
                freeze=conf.freeze_encoder,
                lim_context_sec=conf.context_limit_cpc_sec,
                frame_hz = self.conf.frame_hz
            )
        
        # elif self.conf.encoder_type == "wav2vec2":
        #     from vap.customwav2vec2 import W2V2Transformers
        #     self.encoder = W2V2Transformers(model_type=self.conf.wav2vec_type)
        
        # elif self.conf.encoder_type == "hubert":
        #     from vap.customhubert import HubertEncoder
        #     self.encoder = HubertEncoder(model_type= self.conf.hubert_model)

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
        
        if self.conf.encoder_type == "wav2vec2":
                
            if self.conf.only_feature_extraction == 1:
                self.decrease_dimension = nn.Linear(512, 256)
            else:
                self.decrease_dimension = nn.Linear(1024, 256)
        
        elif self.conf.encoder_type == "hubert":
            
            if self.conf.hubert_model == "hubert_ja":

                if self.conf.only_feature_extraction == 1:
                    self.decrease_dimension = nn.Linear(512, 256)
                else:
                    self.decrease_dimension = nn.Linear(768, 256)
            
            elif self.conf.hubert_model == "hubert_en_large":
                
                if self.conf.only_feature_extraction == 1:
                    self.decrease_dimension = nn.Linear(512, 256)
                else:
                    self.decrease_dimension = nn.Linear(1024, 256)


        if self.conf.freeze_encoder == 1:
            print('freeze encoder')
            self.encoder.freeze()

    @property
    def horizon_time(self):
        return self.objective.horizon_time

    def encode_audio(self, audio: torch.Tensor) -> Tuple[Tensor, Tensor]:
        assert (
            audio.shape[1] == 2
        ), f"audio VAP ENCODER: {audio.shape} != (B, 2, n_samples)"
        
        x1 = self.encoder(audio[:, :1], only_feature_extractor=self.conf.only_feature_extraction)  # speaker 1
        x2 = self.encoder(audio[:, 1:], only_feature_extractor=self.conf.only_feature_extraction)  # speaker 2
        
        # print(audio[:, :1, 0:320])
        # print(x1[0][0])
        # print(x1[0][0].shape)
        # input("check encoder")
        
        #input("check encoder")
        return x1, x2

    def vad_loss(self, vad_output, vad):
        return F.binary_cross_entropy_with_logits(vad_output, vad)

    # def freeze(self):
    #     for p in self.encoder.parameters():
    #         p.requires_grad_(False)
    #     print(f"Froze {self.__class__.__name__}!")

    @torch.no_grad()
    def probs(
        self,
        waveform: Tensor,
        vad: Optional[Tensor] = None,
        now_lims: List[int] = [0, 1],
        future_lims: List[int] = [2, 3],
    ) -> Dict[str, Tensor]:
        
        out = self(waveform)
        probs = out["logits"].softmax(dim=-1)
        vad = out["vad"].sigmoid()
        
        if self.conf.lid_classify >= 1:
            lid = out["lid"].softmax(dim=-1)

        # Calculate entropy over each projection-window prediction (i.e. over
        # frames/time) If we have C=256 possible states the maximum bit entropy
        # is 8 (2^8 = 256) this means that the model have a one in 256 chance
        # to randomly be right. The model can't do better than to uniformly
        # guess each state, it has learned (less than) nothing. We want the
        # model to have low entropy over the course of a dialog, "thinks it
        # understands how the dialog is going", it's a measure of how close the
        # information in the unseen data is to the knowledge encoded in the
        # training data.
        h = -probs * probs.log2()  # Entropy
        H = h.sum(dim=-1)  # average entropy per frame

        # first two bins
        p_now = self.objective.probs_next_speaker_aggregate(
            probs, from_bin=now_lims[0], to_bin=now_lims[-1]
        )
        p_future = self.objective.probs_next_speaker_aggregate(
            probs, from_bin=future_lims[0], to_bin=future_lims[1]
        )

        ret = {
            "probs": probs,
            "vad": vad,
            "p_now": p_now,
            "p_future": p_future,
            "H": H,
        }

        if self.conf.lid_classify >= 1:
            ret.add({"lid": lid})

        if vad is not None:
            labels = self.objective.get_labels(vad)
            ret["loss"] = self.objective.loss_vap(
                out["logits"], labels, reduction="none"
            )
        return ret

    @torch.no_grad()
    def vad(
        self,
        waveform: Tensor,
        max_fill_silence_time: float = 0.02,
        max_omit_spike_time: float = 0.02,
        vad_cutoff: float = 0.5,
    ) -> Tensor:
        """
        Extract (binary) Voice Activity Detection from model
        """
        vad = (self(waveform)["vad"].sigmoid() >= vad_cutoff).float()
        for b in range(vad.shape[0]):
            # TODO: which order is better?
            vad[b] = vad_fill_silences(
                vad[b], max_fill_time=max_fill_silence_time, frame_hz=self.frame_hz
            )
            vad[b] = vad_omit_spikes(
                vad[b], max_omit_time=max_omit_spike_time, frame_hz=self.frame_hz
            )
        return vad
    
    def forward(
        self,
        waveform: Tensor,
        attention: bool = False,
        lang_info: list = None,
    ) -> Dict[str, Tensor]:
        
        x1, x2 = self.encode_audio(waveform)
        
        # Autoregressive
        o1 = self.ar_channel(x1, attention=attention)  # ["x"]
        o2 = self.ar_channel(x2, attention=attention)  # ["x"]
        out = self.ar(o1["x"], o2["x"], attention=attention)

        # Outputs
        v1 = self.va_classifier(out["x1"])
        v2 = self.va_classifier(out["x2"])
        vad = torch.cat((v1, v2), dim=-1)
        logits = self.vap_head(out["x"])

        ret = {"logits": logits, "vad": vad}

        if attention:
            ret["self_attn"] = torch.stack([o1["attn"], o2["attn"]], dim=1)
            ret["cross_attn"] = out["cross_attn"]
            ret["cross_self_attn"] = out["self_attn"]
        
        return ret

