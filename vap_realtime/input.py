import socket
import pyaudio
import queue
import threading
import soundfile as sf
import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame
import time
from . import util
import numpy as np

def available_mic_devices():
    p = pyaudio.PyAudio()
    device_info = {}
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            device_info[info['index']] = info['name']
    return device_info

class Base:
    FRAME_SIZE = 160
    SAMPLING_RATE = 16000
    def get_audio_data(self):
        raise NotImplementedError
    def start_process(self):
        pass

class Mic(Base):
    def __init__(self, audio_gain=1.0, mic_device_index=0):
        self.p = pyaudio.PyAudio()
        self.audio_gain = audio_gain
        self.mic_device_index = mic_device_index
        self.stream = self.p.open(format=pyaudio.paFloat32,
                                  channels=1,
                                  rate=self.SAMPLING_RATE,
                                  input=True,
                                  output=False,
                                  input_device_index=self.mic_device_index)
        
    def get_audio_data(self):
        d = self.stream.read(self.FRAME_SIZE, exception_on_overflow=False)
        d = np.frombuffer(d, dtype=np.float32)
        d = [float(a) for a in d]
        return d

class Wav(Base):
    def __init__(self, wav_file_path, audio_gain=1.0):
        self.wav_file_path = wav_file_path
        self.audio_gain = audio_gain
        self.raw_wav_queue = queue.Queue()
        data, _ = sf.read(file=self.wav_file_path, dtype='float32')
        for i in range(0, len(data), self.FRAME_SIZE):
            if i + self.FRAME_SIZE > len(data):
                break
            d = data[i:i+self.FRAME_SIZE]
            self.raw_wav_queue.put(d)
        self.wav_queue = queue.Queue()
        
    def _read_wav(self):
        start_time = time.time()
        frame_duration = self.FRAME_SIZE / self.SAMPLING_RATE
        pygame.mixer.init(frequency=16000, size=-16, channels=1, buffer=512)
        sound = pygame.mixer.Sound(self.wav_file_path)
        sound.play()
        frame_count = 0
        while True:
            if self.raw_wav_queue.empty():
                break
            expected_time = start_time + frame_count * frame_duration
            current_time = time.time()
            if current_time < expected_time:
                time.sleep(0.001)
                continue
            data = self.raw_wav_queue.get()
            frame_count += 1
            if data is None:
                continue
            self.wav_queue.put(data)
            
    def start_process(self):
        threading.Thread(target=self._read_wav, daemon=True).start()
        
    def get_audio_data(self):
        return self.wav_queue.get()

class TCPReceiver(Base):
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.wav_queue = queue.Queue()
        
    def _start_server(self):
        while True:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind((self.ip, self.port))
                s.listen(1)
                print('[IN] Waiting for connection of audio input...')
                conn, addr = s.accept()
                print('[IN] Connected by', addr)
                while True:
                    size_recv = 8 * 2 * self.FRAME_SIZE
                    data = conn.recv(size_recv)
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
                    x1 = util.conv_bytearray_2_floatarray(data)
                    self.wav_queue.put(x1)
            except Exception as e:
                print('[IN] Disconnected by', addr)
                print(e)
                continue
            
    def start_process(self):
        threading.Thread(target=self._start_server, daemon=True).start()
        
    def get_audio_data(self):
        return self.wav_queue.get()

class TCPTransmitter(Base):
    def __init__(self, ip, port, audio_gain=1.0, mic_device_index=0):
        self.ip = ip
        self.port = port
        self.p = pyaudio.PyAudio()
        self.audio_gain = audio_gain
        self.mic_device_index = mic_device_index
        self.stream = self.p.open(format=pyaudio.paFloat32,
                                  channels=1,
                                  rate=self.SAMPLING_RATE,
                                  input=True,
                                  output=False,
                                  input_device_index=self.mic_device_index)
    
    def connect_server(self):    
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.ip, self.port))
        print('Connected to the server')

    def _start_client(self):
        
        self.connect_server()
        
        self.is_running = True

        while True:
            
            try:
                d = self.stream.read(self.FRAME_SIZE, exception_on_overflow=False)
                
                if self.audio_gain != 1.0:
                    d = np.frombuffer(d, dtype=np.float32) * self.audio_gain
                else:
                    d = np.frombuffer(d, dtype=np.float32)
                
                d = [float(a) for a in d]
                
                if self.is_running:
                    data_sent = util.conv_floatarray_2_byte(d)
                    self.sock.sendall(data_sent)
            except Exception as e:
                print(e)
                continue
            
    def start_process(self):
        threading.Thread(target=self._start_client, daemon=True).start()