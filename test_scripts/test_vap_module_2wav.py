#!/usr/bin/env python3
"""
VAPモデルのテストスクリプト
WAVファイルを使用してVAPモデルの動作を確認します
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
    global wav1, wav2, p_ns, p_ft
    
    wav_file_path1 = "input/wav_sample/jpn_inoue_16k.wav"
    wav_file_path2 = "input/wav_sample/jpn_sumida_16k.wav"
    
    vap = Vap(
        mode="vap",
        frame_rate=20,
        context_len_sec=2.5,
        mic1=VapInput.Wav(wav_file_path=wav_file_path1),
        mic2=VapInput.Wav(wav_file_path=wav_file_path2),
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
        
        p_ns = np.append(p_ns, result['p_now'][0])
        p_ns = p_ns[-MAX_CONTEXT_LEN:]
        
        p_ft = np.append(p_ft, result['p_future'][0])
        p_ft = p_ft[-MAX_CONTEXT_LEN:]
        
if __name__ == "__main__":
    
    SHOWN_CONTEXT_LEN_SEC = 10
    SAMPLE_RATE = 16000
    SAMPE_VAP_RATE = 20
    
    MAX_CONTEXT_LEN = SAMPE_VAP_RATE * SHOWN_CONTEXT_LEN_SEC
    MAX_CONTEXT_WAV_LEN = SAMPLE_RATE * SHOWN_CONTEXT_LEN_SEC
    
    def animate(frame_index):
        
        ax1_sub.set_data(time_x1, wav1)
        ax2_sub.set_data(time_x2, wav2)
        
        #ax3.cla()
        ax3_sub1 = ax3.fill_between(
            time_x3,
            y1=0.5,
            y2=p_ns,
            where=p_ns > 0.5,
            #alpha=alpha_ns,
            color='y',
            #label=label[0],
        )
        ax3_sub2 = ax3.fill_between(
            time_x3,
            y1=p_ns,
            y2=0.5,
            where=p_ns < 0.5,
            #alpha=alpha_ns,
            color='b',
            #label=label[0],
        )

        ax4_sub1 = ax4.fill_between(
            time_x4,
            y1=0.5,
            y2=p_ft,
            where=p_ft > 0.5,
            #alpha=alpha_ns,
            color='y',
            #label=label[0],
        )
        ax4_sub2 = ax4.fill_between(
            time_x4,
            y1=p_ft,
            y2=0.5,
            where=p_ft < 0.5,
            #alpha=alpha_ns,
            color='b',
            #label=label[0],
        )


        return ax1_sub, ax2_sub, ax3_sub1, ax3_sub2, ax4_sub1, ax4_sub2

    root = tkinter.Tk()
    root.wm_title("Real-time VAP demo")

    fig, ax = plt.subplots(4, 1, tight_layout=True, figsize=(18,13))

    # Wave 1
    ax1 = ax[0]
    ax1.set_title('Input waveform 1')
    ax1.set_xlabel('Time [s]')

    time_x1 = np.linspace(-SHOWN_CONTEXT_LEN_SEC, 0, MAX_CONTEXT_WAV_LEN)
    wav1 = np.zeros(len(time_x1))

    ax1.set_ylim(-1, 1)
    ax1.set_xlim(-SHOWN_CONTEXT_LEN_SEC, 0)
    ax1_sub, = ax1.plot(time_x1, wav1, c='y')

    # Wave 2
    ax2 = ax[1]
    ax2.set_title('Input waveform 2')
    ax2.set_xlim(-SHOWN_CONTEXT_LEN_SEC, 0)
    ax2.set_xlabel('Time [s]')

    time_x2 = np.linspace(-SHOWN_CONTEXT_LEN_SEC, 0, MAX_CONTEXT_WAV_LEN)
    wav2 = np.zeros(len(time_x2))

    ax2.set_ylim(-1, 1)
    ax2_sub, = ax2.plot(time_x2, wav2, c='b')

    # p_now
    ax3 = ax[2]
    ax3.set_title('Output p_now (short-term turn-taking prediction)')
    ax3.set_xlabel('Sample')
    ax3.set_xlim(0, MAX_CONTEXT_LEN)
    ax3.set_ylim(0, 1)
    time_x3 = np.linspace(0, MAX_CONTEXT_LEN, MAX_CONTEXT_LEN)
    p_ns = np.ones(len(time_x3)) * 0.5
    ax3_sub1 = ax3.fill_between(
        time_x3,
        y1=0.5,
        y2=p_ns,
        where=p_ns > 0.5,
        #alpha=alpha_ns,
        color='y',
        #label=label[0],
    )
    ax3_sub2 = ax3.fill_between(
        time_x3,
        y1=p_ns,
        y2=0.5,
        where=p_ns < 0.5,
        #alpha=alpha_ns,
        color='b',
        #label=label[0],
    )

    # p_future
    ax4 = ax[3]
    ax4.set_title('Output p_future (long-term turn-taking prediction)')
    ax3.set_xlabel('Sample')
    ax4.set_xlim(0, MAX_CONTEXT_LEN)
    ax4.set_ylim(0, 1)
    time_x4 = np.linspace(0, MAX_CONTEXT_LEN, MAX_CONTEXT_LEN)
    p_ft = np.ones(len(time_x4)) * 0.5
    ax4_sub1 = ax4.fill_between(
        time_x4,
        y1=0.5,
        y2=p_ft,
        where=p_ft > 0.5,
        #alpha=alpha_ns,
        color='y',
        #label=label[0],
    )
    ax4_sub2 = ax4.fill_between(
        time_x4,
        y1=p_ft,
        y2=0.5,
        where=p_ft < 0.5,
        #alpha=alpha_ns,
        color='b',
        #label=label[0],
    )

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