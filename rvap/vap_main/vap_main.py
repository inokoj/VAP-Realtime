import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional

from encoder import EncoderCPC
from modules import GPT, GPTStereo
from objective import ObjectiveVAP

# from wav import WavLoadForVAP
import time
import threading

import numpy as np

import socket
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

    def load_encoder(self, cpc_model):
        
        # Audio Encoder
        #if self.conf.encoder_type == "cpc":
        self.encoder1 = EncoderCPC(
            load_pretrained=True if self.conf.load_pretrained == 1 else False,
            freeze=self.conf.freeze_encoder,
            cpc_model=cpc_model
        )
        self.encoder1 = self.encoder1.eval()
        #print(self.encoder1)
        #self.encoder1 = self.encoder1.half()
        
        self.encoder2 = EncoderCPC(
            load_pretrained=True if self.conf.load_pretrained == 1 else False,
            freeze=self.conf.freeze_encoder,
            cpc_model=cpc_model
        )

        self.encoder2 = self.encoder2.eval()
        #self.encoder2 = self.encoder2.half()
        
        if self.conf.freeze_encoder == 1:
            print('freeze encoder')
            self.encoder1.freeze()
            self.encoder2.freeze()

    @property
    def horizon_time(self):
        return self.objective.horizon_time

    def encode_audio(self, audio1: torch.Tensor, audio2: torch.Tensor) -> Tuple[Tensor, Tensor]:
        
        #
        # audio: (B, 2, n_samples)
        #
        # assert (
        #     audio.shape[1] == 2
        # ), f"audio VAP ENCODER: {audio.shape} != (B, 2, n_samples)"
        
        x1 = self.encoder1(audio1)  # speaker 1
        x2 = self.encoder2(audio2)  # speaker 2
        
        #print(x1.shape)
        #input('Press Enter to continue...')
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

class VAPRealTime():
    
    BINS_P_NOW = [0, 1]
    BINS_PFUTURE = [2, 3]
    
    CALC_PROCESS_TIME_INTERVAL = 100
        
    def __init__(self, vap_model, cpc_model):
        
        conf = VapConfig()
        self.vap = VapGPT(conf)

        sd = torch.load(vap_model, map_location=torch.device('cpu'))
        self.vap.load_encoder(cpc_model=cpc_model)
        self.vap.load_state_dict(sd, strict=False)
        
        # The downsampling parameters are not loaded by "load_state_dict"
        self.vap.encoder1.downsample[1].weight = nn.Parameter(sd['encoder.downsample.1.weight'])
        self.vap.encoder1.downsample[1].bias = nn.Parameter(sd['encoder.downsample.1.bias'])
        self.vap.encoder1.downsample[2].ln.weight = nn.Parameter(sd['encoder.downsample.2.ln.weight'])
        self.vap.encoder1.downsample[2].ln.bias = nn.Parameter(sd['encoder.downsample.2.ln.bias'])
        
        self.vap.encoder2.downsample[1].weight = nn.Parameter(sd['encoder.downsample.1.weight'])
        self.vap.encoder2.downsample[1].bias = nn.Parameter(sd['encoder.downsample.1.bias'])
        self.vap.encoder2.downsample[2].ln.weight = nn.Parameter(sd['encoder.downsample.2.ln.weight'])
        self.vap.encoder2.downsample[2].ln.bias = nn.Parameter(sd['encoder.downsample.2.ln.bias'])

        self.vap = self.vap.eval()

        # Context length of the audio embeddings (depends on frame rate)
        self.AUDIO_CONTEXT_LIM = 50
        
        self.audio_contenxt_lim_sec = 3
        self.frame_hz = 20
        self.sampling_rate = 16000
        self.frame_contxt_padding = 320 # Independe from frame size
        
        # Frame size
        # 20Hz -> 320 + 800 samples
        # 50Hz -> 320 + 320 samples
        self.audio_frame_size = self.sampling_rate // self.frame_hz + self.frame_contxt_padding
        
        self.current_x1_audio = []
        self.current_x2_audio = []
        
        self.result_p_now = 0.
        self.result_p_future = 0.
        self.result_last_time = -1

        self.process_time_abs = -1

        self.e1_context = []
        self.e2_context = []
        
        self.list_process_time_context = []
    
    def process_vap(self, x1, x2):
        
        # Frame size
        # 20Hz -> 320 + 800 samples
        # 50Hz -> 320 + 320 samples
        
        time_start = time.time()
        
        # Save the current audio data
        self.current_x1_audio = x1[self.frame_contxt_padding:]
        self.current_x2_audio = x2[self.frame_contxt_padding:]
        
        with torch.no_grad():
            
            x1_ = torch.tensor([[x1]], dtype=torch.float32)
            x2_ = torch.tensor([[x2]], dtype=torch.float32)

            e1, e2 = self.vap.encode_audio(x1_, x2_)
            
            self.e1_context.append(e1)
            self.e2_context.append(e2)

            if len(self.e1_context) > self.AUDIO_CONTEXT_LIM:
                self.e1_context = self.e1_context[-self.AUDIO_CONTEXT_LIM:]
            if len(self.e2_context) > self.AUDIO_CONTEXT_LIM:
                self.e2_context = self.e2_context[-self.AUDIO_CONTEXT_LIM:]
            
            x1_ = torch.cat(self.e1_context, dim=1)
            x2_ = torch.cat(self.e2_context, dim=1)
            
            o1 = self.vap.ar_channel(x1_, attention=False)
            o2 = self.vap.ar_channel(x2_, attention=False)
            out = self.vap.ar(o1["x"], o2["x"], attention=False)

            # Outputs
            logits = self.vap.vap_head(out["x"])
            probs = logits.softmax(dim=-1)
            
            p_now = self.vap.objective.probs_next_speaker_aggregate(
                probs,
                from_bin=self.BINS_P_NOW[0],
                to_bin=self.BINS_P_NOW[-1]
            )
            
            p_future = self.vap.objective.probs_next_speaker_aggregate(
                probs,
                from_bin=self.BINS_PFUTURE[0],
                to_bin=self.BINS_PFUTURE[1]
            )
            
            self.result_p_now = p_now.tolist()[0][-1]
            self.result_p_future = p_future.tolist()[0][-1]
            self.result_last_time = time.time()
            
            time_process = time.time() - time_start
            
            # Calculate the average encoding time
            self.list_process_time_context.append(time_process)
            
            if len(self.list_process_time_context) > self.CALC_PROCESS_TIME_INTERVAL:
                ave_proc_time = np.average(self.list_process_time_context)
                print('[VAP] Average processing time: %.5f [sec]' % ave_proc_time)
                self.list_process_time_context = []
            
            self.process_time_abs = time.time()
                    

def proc_serv_out(list_socket_out, port_number=50008):
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        
        s.bind(('127.0.0.1', port_number))
        s.listen(1)

        while True:
            conn, addr = s.accept()
            print('[OUT] Connected by', addr)
            conn.setblocking(False)
            conn.settimeout(0)
            list_socket_out.append(conn)
            
            print('[OUT] Current client num = %d' % len(list_socket_out))

def proc_serv_in(port_number, vap):
    
    FRAME_SIZE_INPUT = 160
    FRAME_SIZE_VAP = 800
    FRAME_SAVE_LAST = 320
    
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(('127.0.0.1', port_number))
            s.listen(1)
        
            print('[IN] Waiting for connection of audio input...')
            conn, addr = s.accept()
            print('[IN] Connected by', addr)
            
            current_x1 = np.zeros(FRAME_SAVE_LAST)
            current_x2 = np.zeros(FRAME_SAVE_LAST)
            
            while True:
                
                # Float (4 byte) x 2 persons x 160 samples (0.01 sec)
                size_recv = 8 * 2 * FRAME_SIZE_INPUT
                data = conn.recv(size_recv)

                # Continue to receive data until the size of the data is 
                # equal to the size of the data to be received
                if len(data) < size_recv:
                    while True:
                        data_ = conn.recv(size_recv)
                        if len(data_) == 0:
                            break
                        data += data_
                        if len(data) == size_recv:
                            break
                
                if len(data) == 0:
                    break
                
                x1, x2 = util.conv_bytearray_2_2floatarray(data)
                
                current_x1 = np.concatenate([current_x1, x1])
                current_x2 = np.concatenate([current_x2, x2])
                
                # Continue to receive data until the size of the data is
                # less that the size of the VAP frame
                if len(current_x1) < FRAME_SIZE_VAP + FRAME_SAVE_LAST:
                    continue
                
                vap.process_vap(current_x1, current_x2)
                
                # Save the last 320 samples
                current_x1 = current_x1[-FRAME_SAVE_LAST:]
                current_x2 = current_x2[-FRAME_SAVE_LAST:]
                
        except Exception as e:
            print('[IN] Disconnected by', addr)
            print(e)
            continue

def proc_serv_out_dist(list_socket_out, vap):
    
    previous_time = vap.process_time_abs
    
    while True:
        
        if previous_time == vap.process_time_abs:
            time.sleep(1E-5)
            continue
        
        previous_time = vap.process_time_abs
        
        t = copy.copy(vap.result_last_time)
        x1 = copy.copy(vap.current_x1_audio)
        x2 = copy.copy(vap.current_x2_audio)
        p_now = copy.copy(vap.result_p_now)
        p_future = copy.copy(vap.result_p_future)
        
        vap_result = {
            "t": t,
            "x1": x1, "x2": x2,
            "p_now": p_now, "p_future": p_future
        }
        
        data_sent = util.conv_vapresult_2_bytearray(vap_result)
        sent_size = len(data_sent)
        data_sent_all = sent_size.to_bytes(4, 'little') + data_sent
        
        for conn in list_socket_out:
            try:
                if conn.fileno() != -1:
                    conn.sendall(data_sent_all)
            except:
                print('[OUT] Disconnected by', conn.getpeername())
                list_socket_out.remove(conn)
                continue

    #previous_time = result_time_stamp[-1]


if __name__ == "__main__":
    
    #checkpoint_dict = './model/kankou_cpc_conlim_50/VapGPT_50Hz_ad20s_134_cpc-epoch6-val_2.02414.ckpt'
    #model.load_from_checkpoint(checkpoint)
    
    # Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vap_model", type=str, default='../asset/vap/vap_state_dict_20hz_jpn.pt')
    parser.add_argument("--cpc_model", type=str, default='../asset/cpc/60k_epoch4-d0f474de.pt')
    parser.add_argument("--port_num_in", type=int, default=50007)
    parser.add_argument("--port_num_out", type=int, default=50008)
    # parser.add_argument("--input_wav_left", type=str, default='wav/101_1_2-left-ope.wav')
    # parser.add_argument("--input_wav_right", type=str, default='wav/101_1_2-right-cus.wav')
    # parser.add_argument("--play_wav_stereo", type=str, default='wav/101_1_2.wav')
    args = parser.parse_args()

    # global current_x1, current_x2
    # global result_p_now, result_p_future, result_vad, result_time_stamp
    # result_p_now = []
    # result_p_future = []
    # result_vad = []
    # result_time_stamp = [0]
    
    wait_input = True

    vap = VAPRealTime(args.vap_model, args.cpc_model)
    
    list_socket_out = []

    # Start the server to receive the connection for sending the VAP results
    t_server_out_connect = threading.Thread(target=proc_serv_out, args=(list_socket_out, args.port_num_out,))
    t_server_out_connect.setDaemon(True)
    t_server_out_connect.start()

    # Check the connection and processing of the VAP and then send the VAP results
    t_server_out_distribute = threading.Thread(target=proc_serv_out_dist, args=(list_socket_out, vap))
    t_server_out_distribute.setDaemon(True)
    t_server_out_distribute.start()

    # This process must be run in the main thread
    proc_serv_in(args.port_num_in, vap)
    
    # Continue unitl pressind "Ctrl+c"
    print('Press Ctrl+c to stop')
    while True:
        time.sleep(1E-5)
        pass
    
   