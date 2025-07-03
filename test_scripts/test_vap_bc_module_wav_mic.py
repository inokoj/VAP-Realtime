#!/usr/bin/env python3
"""
VAPバックチャンネルモデルのテストスクリプト
WAVファイルとマイクを使用してVAPバックチャンネルモデルの動作を確認し、
p_bc_react, p_bc_emoを可視化します。
"""

import sys
import os
import numpy as np
import tkinter
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.pyplot as plt
from matplotlib import animation
import threading

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vap_realtime import Vap, VapInput

def test_vap_with_gui():
    global wav1, wav2, p_bc_react, p_bc_emo
    
    wav_file_path1 = "input/wav_sample/jpn_sumida_16k.wav"
    
    vap = Vap(
        mode="bc",
        frame_rate=10,
        context_len_sec=5,
        mic1=VapInput.Mic(mic_device_index=0),
        mic2=VapInput.Wav(wav_file_path=wav_file_path1),
        device="cpu"
    )
    
    vap.start_process()
    
    while True:
        result = vap.get_result()
        
        x1 = result['x1']
        wav1 = np.append(wav1, x1)
        wav1 = wav1[-MAX_CONTEXT_WAV_LEN:]
        
        x2 = result['x2']
        wav2 = np.append(wav2, x2)
        wav2 = wav2[-MAX_CONTEXT_WAV_LEN:]
        
        # バックチャンネル出力
        p_bc_react = np.append(p_bc_react, result['p_bc_react'][0])
        p_bc_react = p_bc_react[-MAX_CONTEXT_LEN:]
        p_bc_emo = np.append(p_bc_emo, result['p_bc_emo'][0])
        p_bc_emo = p_bc_emo[-MAX_CONTEXT_LEN:]
        
if __name__ == "__main__":
    
    SHOWN_CONTEXT_LEN_SEC = 10
    SAMPLE_RATE = 16000
    SAMPE_VAP_RATE = 20
    
    MAX_CONTEXT_LEN = SAMPE_VAP_RATE * SHOWN_CONTEXT_LEN_SEC
    MAX_CONTEXT_WAV_LEN = SAMPLE_RATE * SHOWN_CONTEXT_LEN_SEC
    
    def animate(frame_index):
        ax1_sub.set_data(time_x1, wav1)
        ax2_sub.set_data(time_x2, wav2)
        
        ax3_sub1 = ax3.fill_between(
            time_x3,
            y1=0.0,
            y2=p_bc_react,
            color='tomato',
        )
        ax3.axhline(y=0.5, color='black', linestyle='--')

        ax4_sub1 = ax4.fill_between(
            time_x4,
            y1=0.0,
            y2=p_bc_emo,
            color='violet',
        )
        ax4.axhline(y=0.5, color='black', linestyle='--')

        return ax1_sub, ax2_sub, ax3_sub1, ax4_sub1

    root = tkinter.Tk()
    root.wm_title("Real-time VAP BC demo (WAV + Mic input)")

    fig, ax = plt.subplots(4, 1, tight_layout=True, figsize=(18,13))

    # Wave 1
    ax1 = ax[0]
    ax1.set_title('Target (Channel 1: WAV)')
    ax1.set_xlabel('Time [s]')

    time_x1 = np.linspace(-SHOWN_CONTEXT_LEN_SEC, 0, MAX_CONTEXT_WAV_LEN)
    wav1 = np.zeros(len(time_x1))

    ax1.set_ylim(-1, 1)
    ax1.set_xlim(-SHOWN_CONTEXT_LEN_SEC, 0)
    ax1_sub, = ax1.plot(time_x1, wav1, c='y')

    # Wave 2
    ax2 = ax[1]
    ax2.set_title('Input waveform (Channel 2: Mic)')
    ax2.set_xlim(-SHOWN_CONTEXT_LEN_SEC, 0)
    ax2.set_xlabel('Time [s]')

    time_x2 = np.linspace(-SHOWN_CONTEXT_LEN_SEC, 0, MAX_CONTEXT_WAV_LEN)
    wav2 = np.zeros(len(time_x2))

    ax2.set_ylim(-1, 1)
    ax2_sub, = ax2.plot(time_x2, wav2, c='b')

    # p_bc_react
    ax3 = ax[2]
    ax3.set_title('Output p_bc_react (reactive backchannel prediction)')
    ax3.set_xlabel('Sample')
    ax3.set_xlim(0, MAX_CONTEXT_LEN)
    ax3.set_ylim(0, 1)
    time_x3 = np.linspace(0, MAX_CONTEXT_LEN, MAX_CONTEXT_LEN)
    p_bc_react = np.ones(len(time_x3)) * 0.0
    ax3_sub1 = ax3.fill_between(
        time_x3,
        y1=0.0,
        y2=p_bc_react,
        color='tomato',
    )
    ax3.axhline(y=0.5, color='black', linestyle='--')

    # p_bc_emo
    ax4 = ax[3]
    ax4.set_title('Output p_bc_emo (emotional backchannel prediction)')
    ax3.set_xlabel('Sample')
    ax4.set_xlim(0, MAX_CONTEXT_LEN)
    ax4.set_ylim(0, 1)
    time_x4 = np.linspace(0, MAX_CONTEXT_LEN, MAX_CONTEXT_LEN)
    p_bc_emo = np.ones(len(time_x4)) * 0.0
    ax4_sub1 = ax4.fill_between(
        time_x4,
        y1=0.0,
        y2=p_bc_emo,
        color='violet',
    )
    ax4.axhline(y=0.5, color='black', linestyle='--')

    ani = animation.FuncAnimation(
        fig,
        animate,
        interval=250,
        blit=True
    )

    # matplotlib を GUI(Tkinter) に追加する
    canvas = FigureCanvasTkAgg(fig, master=root)
    toolbar = NavigationToolbar2Tk(canvas, root)
    canvas.get_tk_widget().pack()
    
    thread_vap = threading.Thread(target=test_vap_with_gui)
    thread_vap.setDaemon(True)
    thread_vap.start()
    
    tkinter.mainloop()