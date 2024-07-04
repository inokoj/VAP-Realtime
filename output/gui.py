
import tkinter
from matplotlib.backends.backend_tkagg import (
	FigureCanvasTkAgg, NavigationToolbar2Tk)

import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import animation

import threading
import argparse
import numpy as np
import socket

import rvap.common.util as util

def key_event(e, server_ip='127.0.0.1', port_num=50009):
    
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((server_ip, port_num))
    
    if str(e.char).strip() == 'p':
        try:
            s.send(b'p\n')
            print('send p')
        except:
            pass
    
    if str(e.char).strip() == 'r':
        try:
            s.send(b'r\n')
            print('send r')
        except:
            pass
            
def process_client(server_ip='127.0.0.1', port_num=50008):
    
    global wav1, time_x1, wav2, time_x2
    global p_ns, time_x3
    global p_ft, time_x4
    #global wavPlay

    #list_socket = []
    
    # Start to connect to the server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock_server:
        sock_server.connect((server_ip, port_num))
        #sock_server.settimeout(0.1)
        import time
        
        while True:
            try:
                #time_start = time.time()
                
                data_size = sock_server.recv(4)
                size = int.from_bytes(data_size, 'little')
                
                data = sock_server.recv(size)
                while len(data) < size:
                    data += sock_server.recv(size - len(data))
               
                vap_result = util.conv_bytearray_2_vapresult(data)
                
                x1 = vap_result['x1']
                wav1 = np.append(wav1, x1)
                wav1 = wav1[-MAX_CONTEXT_WAV_LEN:]

                x2 = vap_result['x2']
                wav2 = np.append(wav2, x2)
                wav2 = wav2[-MAX_CONTEXT_WAV_LEN:]

                #print(data['p_now'])
                p_ns = np.append(p_ns, vap_result['p_now'][0])
                p_ns = p_ns[-MAX_CONTEXT_LEN:]

                p_ft = np.append(p_ft, vap_result['p_future'][0])
                p_ft = p_ft[-MAX_CONTEXT_LEN:]
                
                #print('Time elapsed: %.3f' % (time.time() - time_start))

            except Exception as e:
                print('Disconnected from the server')
                print(e)
                break

SHOWN_CONTEXT_LEN_SEC = 10
SAMPLE_RATE = 16000
SAMPE_VAP_RATE = 20

MAX_CONTEXT_LEN = SAMPE_VAP_RATE * SHOWN_CONTEXT_LEN_SEC
MAX_CONTEXT_WAV_LEN = SAMPLE_RATE * SHOWN_CONTEXT_LEN_SEC

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_ip", type=str, default='127.0.0.1')
    parser.add_argument("--port_num", type=int, default=50008)
    parser.add_argument("--wav_command_server_ip", type=str, default='127.0.0.1')
    parser.add_argument("--wav_command_server_port_num", type=int, default=50009)
    args = parser.parse_args()
    
    sns.set()
    sns.set(font_scale=1.3)

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
    
    root.bind('<Any-KeyPress>',
              lambda event: key_event(
                  event,
                  args.wav_command_server_ip,
                  args.wav_command_server_port_num
                )
            )

    # Start to process the client
    thread_client = threading.Thread(target=process_client)
    thread_client.setDaemon(True)
    thread_client.start()

    # GUIを開始，GUIが表示されている間は処理はここでストップ
    tkinter.mainloop()