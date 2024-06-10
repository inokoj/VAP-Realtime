#
# This sample program is for sending audio data captured from a microphone to the server.
# The VAP model takes two audio signals as input, so the second channel is filled with zeros.
#

import argparse
import socket
import threading
import pyaudio

import numpy as np
import os
import sys
sys.path.append(os.pardir)
import common.util as util

class MicLoaderForVAP:   
    
    FRAME_SIZE = 160
    SAMPLING_RATE = 16000
    
    def __init__(self,
                server_ip='127.0.0.1',
                server_port=50007
        ):

        # Set up the microphone
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paFloat32,
                                  channels=1,
                                  rate=self.SAMPLING_RATE,
                                  input=True,
                                  output=False
                                  )

        self.server_ip = server_ip
        self.server_port = server_port

        print('----------------------------------')
        print('Server IP: {}'.format(self.server_ip))
        print('Server Port: {}'.format(self.server_port))
        print('----------------------------------')
    
    def connect_server(self):    
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.server_ip, self.server_port))
        print('Connected to the server')

    def callback(self, in_data, frame_count, time_info, status):
        print(in_data)
        return (in_data, pyaudio.paContinue)
    
    def start(self):
        
        self.connect_server()
        x2_vap = [0.0] * self.FRAME_SIZE

        while True:
            
            d = self.stream.read(self.FRAME_SIZE)
            d = [float(a) for a in np.frombuffer(d, dtype=np.float32)]
            data_sent = util.conv_2floatarray_2_bytearray(d, x2_vap)
            self.sock.sendall(data_sent)
            
def process_server_command(server_ip='127.0.0.1', port_number=50009, loader=None):
    
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind((server_ip, port_number))
            s.listen(1)
            
            print('[COMMAND] Waiting for connection of command...')
            conn, addr = s.accept()
            print('[COMMAND] Connected by', addr)
            
            while True:
                command = conn.recv(1).decode('utf-8')
                
                if len(command) == 0:
                    break
                
                if command == 'p':
                    print('[COMMAND] Pause')
                    loader.stream.stop_stream
                elif command == 'r':
                    print('[COMMAND] Resume')
                    loader.stream.start_stream()
                else:
                    print('[COMMAND] Unknown command')
        except Exception as e:
            print(e)
            continue
    
if __name__ == '__main__':
    
    # Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_ip", type=str, default='127.0.0.1')
    parser.add_argument("--port_num", type=int, default=50007)
    parser.add_argument("--command_server_port_num", type=int, default=50009)
    args = parser.parse_args()
    
    # Load wav files and send data to the server
    loader = MicLoaderForVAP(
        server_ip=args.server_ip,
        server_port=args.port_num
    )

    # Start to process the client
    command_server_ip = '127.0.0.1'
    thread_server = threading.Thread(target=process_server_command, args=(command_server_ip, args.command_server_port_num))
    thread_server.setDaemon(True)
    thread_server.start()
    
    loader.start()