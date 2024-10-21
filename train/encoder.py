import torch
import torch.nn as nn
import einops
import os
import numpy as np
from encoder_components import load_CPC, get_cnn_layer

import time

class EncoderCPC(nn.Module):
    """
    Encoder: waveform -> h
    pretrained: default='cpc'

    A simpler version of the Encoder
    check paper (branch) version to see other encoders...
    """

    def __init__(self, cpc_model_pt='', load_pretrained=True, freeze=True, lim_context_sec: float=-1, frame_hz: int=50):
        super().__init__()
        self.sample_rate = 16000
        self.encoder = load_CPC(cpc_model_pt, load_pretrained)
        self.output_dim = self.encoder.gEncoder.conv4.out_channels
        self.dim = self.output_dim
        
        #
        # For real-time processing
        #
        self.frame_hz = frame_hz
        kernel_size = int((16000 / 50 / 160) * 2.5)
        stride = int(self.sample_rate / self.frame_hz / 160)

        kernel_size = int(100 / self.frame_hz)
        stride = int(100 / self.frame_hz)
        
        self.downsample = get_cnn_layer(
            dim=self.output_dim,
            kernel=[kernel_size],
            stride=[stride],
            dilation=[1],
            activation="GELU",
        )

        #
        # Original VAP
        #
        # self.downsample = get_cnn_layer(
        #     dim=self.output_dim,
        #     kernel=[5],
        #     stride=[2],
        #     dilation=[1],
        #     activation="GELU",
        # )
        
        # self.downsample_ratio = 320
        
        # 入力の長さを制限
        self.lim_context_sec = lim_context_sec

        # 1000が割り切れるようにする
        self.STEP_SIZE_BY_CONTEXT_LIM = {
            15: 10,
            10: 10,
            5: 25,
            3: 50,
            2: 50,
            1: 100,
        }

        if freeze:
            self.freeze()
        else:
            # CNNs are frozen by default
            self.unfreeze()

    def get_default_conf(self):
        return {""}

    def freeze(self):
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        print(f"Froze {self.__class__.__name__}!")

    def unfreeze(self):
        
        # Unfreeze only the autoregressive part (except the CNN layers)
        self.freeze()
        self.encoder.gAR.requires_grad_(True)
        
        #for p in self.encoder.parameters():
        #     p.requires_grad_(True)
        print(f"Trainable {self.__class__.__name__}!")

    def forward(self, waveform, only_feature_extractor: int = 0):
        
        if waveform.ndim < 3:
            waveform = waveform.unsqueeze(1)  # channel dim

        # Backwards using only the encoder encounters:
        # ---------------------------------------------------
        # RuntimeError: one of the variables needed for gradient computation
        # has been modified by an inplace operation:
        # [torch.FloatTensor [4, 256, 1000]], which is output 0 of ReluBackward0, is at version 1;
        # expected version 0 instead. Hint: enable anomaly detection to find
        # the operation that failed to compute its gradient, with
        # torch.autograd.set_detect_anomaly(True).
        # HOWEVER, if we feed through encoder.gAR we do not encounter that problem...

        #print(self.lim_context_sec)
        if self.lim_context_sec < 0:
            z = self.encoder.gEncoder(waveform)
            z = einops.rearrange(z, "b c n -> b n c")
            z = self.encoder.gAR(z)
            z = self.downsample(z)
    
        # print(waveform.shape)
        # print(z.shape)

        # 入力の長さを制限（処理時間がかかる）
        if self.lim_context_sec > 0:
            
            FRAME_PER_FEATURE = 320
            DIM_FEATURE = 256
            num_feature = int(waveform.shape[2] / FRAME_PER_FEATURE) # 20sec -> 1000
            lim_context_n = int(self.lim_context_sec * self.sample_rate)
            z = np.zeros((waveform.shape[0], num_feature, DIM_FEATURE))

            # キャッシュにデータがあるか確認
            cached_idx = []
            for b in range(waveform.shape[0]):
                
                z_hash = self.hash_tensor(waveform[b,:,:])
                tensor_path = f"/n/work1/inoue/temp/%dsec/z%s.pt" % (self.lim_context_sec, z_hash)

                if os.path.exists(tensor_path):
                    try:
                        z_ = torch.load(tensor_path)
                    except:
                        continue

                    z[b,:,:] = z_.cpu().clone().detach()
                    del z_
                    cached_idx.append(b)

            # z_hash = self.hash_tensor(waveform)
            # tensor_path = f"/n/work1/inoue/temp/z_{z_hash}-{self.lim_context_sec}.pt"

            # if os.path.exists(tensor_path):
            #     print("load z")
            #     z = torch.load(tensor_path, torch.device("cuda:0"))
            #     print('z.shape', z.shape)

            batch_size = waveform.shape[0]
            
            if len(cached_idx) < batch_size:
                print("z is not cached\t(num_cached={}/{})".format(len(cached_idx), waveform.shape[0]))
            else:
                print("z is cached\t(num_cached={}/{})".format(len(cached_idx), waveform.shape[0]))
            
            
            if len(cached_idx) < batch_size:

                step_size = self.STEP_SIZE_BY_CONTEXT_LIM[self.lim_context_sec]
                for i in range(0, num_feature, step_size):
                    
                    # measure processing time
                    time_start = time.time()
                    
                    waveform_ = None

                    for j in range(step_size):
                        
                        start_idx = max((i+j+1)*FRAME_PER_FEATURE-lim_context_n, 0)
                        end_idx = min((i+j+1)*FRAME_PER_FEATURE, waveform.shape[2])
                        
                        for b in range(batch_size):

                            w_ = waveform[b, :, start_idx:end_idx].clone().detach()

                            # padding
                            if w_.shape[1] < lim_context_n:
                                w_ = torch.cat((torch.zeros((w_.shape[0], lim_context_n-w_.shape[1]), device=w_.device), w_), dim=1)

                            if waveform_ == None:
                                waveform_ = w_
                            else:
                                waveform_ = torch.cat((waveform_, w_), dim=0)
                    
                    waveform_ = waveform_.unsqueeze(1)  # channel dim
                    # print(waveform_.shape)
                    # input()
                        
                    # time_end = time.time()
                    # print('waveform_ time', time_end - time_start)    
                    
                    # print(waveform_.shape)
                    # input()

                    #time_start = time.time()

                    with torch.no_grad():
                        z_ = self.encoder.gEncoder(waveform_)
                        z_ = einops.rearrange(z_, "b c n -> b n c")
                        z_ = self.encoder.gAR(z_)
                        z_ = self.downsample(z_)
                    
                    # time_end = time.time()
                    # print('z_ time', time_end - time_start)
                    # #print(z_.shape)
                    # input()

                    # print('z_ shape', z_.shape)
                    # input()

                    time_start = time.time()

                    #input()

                    batch_size = waveform.shape[0]
                    idx_copied = 0
                    for j in range(step_size):
                        for b in range(batch_size):
                            
                            if b in cached_idx:
                                continue
                            
                            z[b,i+j,:] = z_[idx_copied,-1,:].to('cpu').detach().numpy().copy()
                            #print(idx_copied)
                            idx_copied += 1

                    del z_, waveform_, w_

            z = torch.from_numpy(z.astype(np.float32))
            z = z.cuda()
            
            # batchごとにzを保存
            for b in range(waveform.shape[0]):
                
                z_hash = self.hash_tensor(waveform[b,:,:])
                tensor_path = f"/n/work1/inoue/temp/%dsec/z%s.pt" % (self.lim_context_sec, z_hash)

                # Hash値をファイル名としてzのデータを保存
                #print(z[b,:,:].shape)
                if os.path.exists(tensor_path) == False:
                    torch.save(z[b,:,:], tensor_path)
                #print("save z as ", tensor_path)
        
        return z

    def hash_tensor(self, tensor):
        return hash(tuple(tensor.reshape(-1).tolist()))
