#
# This sample program is for sending two wav files to the server.
# This also will play the mixed audio file.
#

import soundfile as sf
import queue
import pygame
import argparse
import time
import socket
from pydub import AudioSegment
import threading

import rvap.common.util as util

class WavLoaderForVAP:   
    
    FRAME_SIZE = 160
    SAMPLING_RATE = 16000
    
    def __init__(self,
                filename_user1,
                filename_user2,
                filename_mix=None,
                server_ip='127.0.0.1',
                server_port=50007
        ):
        
        # Create mixed audio file using pydub
        if filename_mix is None:
            sound1 = AudioSegment.from_file(filename_user1)
            sound2 = AudioSegment.from_file(filename_user2)
            output = sound1.overlay(sound2, position=0)
            filename_mix = 'temp_mixed_sounds.wav'
            output.export(filename_mix, format="wav")
            print('Mixed audio file is created as {}'.format(filename_mix))
        
        self.filename1 = filename_user1
        self.filename2 = filename_user2
        self.filename_mix = filename_mix

        self.server_ip = server_ip
        self.server_port = server_port

        print('----------------------------------')
        print('Server IP: {}'.format(self.server_ip))
        print('Server Port: {}'.format(self.server_port))
        print('Input wav file 1: {}'.format(self.filename1))
        print('Input wav file 2: {}'.format(self.filename2))
        print('Mixed wav file: {}'.format(self.filename_mix))
        print('----------------------------------')
    
    def connect_server(self):    
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.server_ip, self.server_port))
        print('Connected to the server')

    def start(self):
        
        self.connect_server()

        self.queue1 = queue.Queue()
        self.queue2 = queue.Queue()

        # Load audio files
        data, _ = sf.read(file=self.filename1, dtype='float32')
        for i in range(0, len(data), self.FRAME_SIZE):
            if i + self.FRAME_SIZE > len(data):
                break
            d = data[i:i+self.FRAME_SIZE]
            self.queue1.put(d)
        
        data, _ = sf.read(file=self.filename2, dtype='float32')
        for i in range(0, len(data), self.FRAME_SIZE):
            if i + self.FRAME_SIZE > len(data):
                break
            d = data[i:i+self.FRAME_SIZE]
            self.queue2.put(d)

        done_millisec = 0

        # start play
        pygame.mixer.init()
        pygame.mixer.music.load(self.filename_mix)
        pygame.mixer.music.play(1)

        print('Start playing wav files')

        # playing and send loop
        while True:
            
            if self.queue1.empty() or self.queue2.empty():
                break

            played_msec = pygame.mixer.music.get_pos()
            
            # Sync with the played audio
            if done_millisec > played_msec:
                time.sleep(1E-10)
                continue

            x1_vap, x2_vap = self.queue1.get(), self.queue2.get()
            done_millisec += int((self.FRAME_SIZE / self.SAMPLING_RATE) * 1000)
            
            if x1_vap is None or x2_vap is None:
                continue
            
            data_sent = util.conv_2floatarray_2_bytearray(x1_vap, x2_vap)
            self.sock.sendall(data_sent)

def process_server_command(server_ip='127.0.0.1', port_number=50009):
    
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
                    pygame.mixer.music.pause()
                elif command == 'r':
                    print('[COMMAND] Resume')
                    pygame.mixer.music.unpause()
                else:
                    print('[COMMAND] Unknown command')
        except Exception as e:
            print(e)
            continue
    
if __name__ == '__main__':
    
    # Argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument("--checkpoint_dict", type=str, default='./model/state_dict_20hz.pt')
    parser.add_argument("--server_ip", type=str, default='127.0.0.1')
    parser.add_argument("--port_num", type=int, default=50007)
    parser.add_argument("--input_wav_left", type=str, default='wav_sample/jpn_inoue_16k.wav')
    parser.add_argument("--input_wav_right", type=str, default='wav_sample/jpn_sumida_16k.wav')
    parser.add_argument("--play_wav_stereo", type=str, default='wav_sample/jpn_mix_16k.wav')
    parser.add_argument("--command_server_port_num", type=int, default=50009)
    args = parser.parse_args()
    
    # Load wav files and send data to the server
    loader = WavLoaderForVAP(
        filename_user1=args.input_wav_left,
        filename_user2=args.input_wav_right,
        filename_mix=args.play_wav_stereo,
        server_ip=args.server_ip,
        server_port=args.port_num
    )
    
    # Start to process the client
    command_server_ip = '127.0.0.1'
    thread_server = threading.Thread(target=process_server_command, args=(command_server_ip, args.command_server_port_num))
    thread_server.setDaemon(True)
    thread_server.start()
    
    loader.start()