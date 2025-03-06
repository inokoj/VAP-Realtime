#!/usr/bin/env python

from __future__ import annotations
import warnings
warnings.filterwarnings('ignore')
import onnxruntime as ort
import soundfile as sf
from enum import Enum
import numpy as np
from typing import List, Tuple, Deque
import argparse
import os
import sys
import importlib.util
from abc import ABC, abstractmethod
from pprint import pprint

class Color(Enum):
    BLACK          = '\033[30m'
    RED            = '\033[31m'
    GREEN          = '\033[32m'
    YELLOW         = '\033[33m'
    BLUE           = '\033[34m'
    MAGENTA        = '\033[35m'
    CYAN           = '\033[36m'
    WHITE          = '\033[37m'
    COLOR_DEFAULT  = '\033[39m'
    BOLD           = '\033[1m'
    UNDERLINE      = '\033[4m'
    INVISIBLE      = '\033[08m'
    REVERSE        = '\033[07m'
    BG_BLACK       = '\033[40m'
    BG_RED         = '\033[41m'
    BG_GREEN       = '\033[42m'
    BG_YELLOW      = '\033[43m'
    BG_BLUE        = '\033[44m'
    BG_MAGENTA     = '\033[45m'
    BG_CYAN        = '\033[46m'
    BG_WHITE       = '\033[47m'
    BG_DEFAULT     = '\033[49m'
    RESET          = '\033[0m'

    def __str__(self):
        return self.value

    def __call__(self, s):
        return str(self) + str(s) + str(Color.RESET)

class AbstractModel(ABC):
    """AbstractModel
    Base class of the model.
    """
    _runtime: str = 'onnx'
    _model_path: str = ''
    _obj_class_score_th: float = 0.35
    _attr_class_score_th: float = 0.70
    _input_shapes: List[List[int]] = []
    _input_names: List[str] = []
    _output_shapes: List[List[int]] = []
    _output_names: List[str] = []

    # onnx/tflite
    _interpreter = None
    _inference_model = None
    _providers = None
    _swap = (2, 0, 1)
    _h_index = 2
    _w_index = 3

    # onnx
    _onnx_dtypes_to_np_dtypes = {
        "tensor(float)": np.float32,
        "tensor(uint8)": np.uint8,
        "tensor(int8)": np.int8,
    }

    # tflite
    _input_details = None
    _output_details = None

    @abstractmethod
    def __init__(
        self,
        *,
        runtime: str = 'onnx',
        model_path: str = '',
        obj_class_score_th: float = 0.35,
        attr_class_score_th: float = 0.70,
        providers: List = [
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': '.',
                    'trt_fp16_enable': True,
                }
            ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    ):
        self._runtime = runtime
        self._model_path = model_path
        self._obj_class_score_th = obj_class_score_th
        self._attr_class_score_th = attr_class_score_th
        self._providers = providers

        # Model loading
        if self._runtime == 'onnx':
            import onnxruntime # type: ignore
            onnxruntime.set_default_logger_severity(3) # ERROR
            session_option = onnxruntime.SessionOptions()
            session_option.log_severity_level = 3
            self._interpreter = \
                onnxruntime.InferenceSession(
                    model_path,
                    sess_options=session_option,
                    providers=providers,
                )
            self._providers = self._interpreter.get_providers()
            print(f'{Color.GREEN("Enabled ONNX ExecutionProviders:")}')
            pprint(f'{self._providers}')

            self._input_shapes = [
                input.shape for input in self._interpreter.get_inputs()
            ]
            self._input_names = [
                input.name for input in self._interpreter.get_inputs()
            ]
            self._input_dtypes = [
                self._onnx_dtypes_to_np_dtypes[input.type] for input in self._interpreter.get_inputs()
            ]
            self._output_shapes = [
                output.shape for output in self._interpreter.get_outputs()
            ]
            self._output_names = [
                output.name for output in self._interpreter.get_outputs()
            ]
            self._model = self._interpreter.run
            self._swap = (0, 1, 2, 3)

        elif self._runtime in ['tflite_runtime', 'tensorflow']:
            if self._runtime == 'tflite_runtime':
                from tflite_runtime.interpreter import Interpreter # type: ignore
                self._interpreter = Interpreter(model_path=model_path)
            elif self._runtime == 'tensorflow':
                import tensorflow as tf # type: ignore
                self._interpreter = tf.lite.Interpreter(model_path=model_path)
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
            self._input_names = [
                input.get('name', None) for input in self._input_details
            ]
            self._input_dtypes = [
                input.get('dtype', None) for input in self._input_details
            ]
            self._output_shapes = [
                output.get('shape', None) for output in self._output_details
            ]
            self._output_names = [
                output.get('name', None) for output in self._output_details
            ]
            self._model = self._interpreter.get_signature_runner()
            self._swap = (0, 1, 2, 3)

    @abstractmethod
    def process_vap(
        self,
        *,
        input_datas: List[np.ndarray],
    ) -> List[np.ndarray]:
        datas = {
            f'{input_name}': input_data \
                for input_name, input_data in zip(self._input_names, input_datas)
        }
        if self._runtime == 'onnx':
            outputs = [
                output for output in \
                    self._model(
                        output_names=self._output_names,
                        input_feed=datas,
                    )
            ]
            return outputs
        elif self._runtime in ['tflite_runtime', 'tensorflow']:
            outputs = [
                output for output in \
                    self._model(
                        **datas
                    ).values()
            ]
            return outputs

class VAPRealTime(AbstractModel):
    def __init__(
        self,
        *,
        runtime: str = 'onnx',
        vap_onnx_model: str = 'vap_state_dict_jp_20hz_2500msec.onnx',
        providers: List = None,
        vap_process_rate: int = 20,
        context_len_sec: float = 5,
    ):
        """

        Parameters
        ----------
        runtime: str
            Runtime for VAP. Default: onnx

        vap_onnx_model: str
            ONNX/TFLite file path for VAP

        providers: List
            Providers for ONNXRuntime.

        vap_process_rate: int

        context_len_sec: float
        """
        super().__init__(
            runtime=runtime,
            model_path=vap_onnx_model,
            providers=providers,
        )
        BINS_P_NOW = [0, 1]
        BINS_PFUTURE = [2, 3]
        CALC_PROCESS_TIME_INTERVAL = 100

        self.audio_contenxt_lim_sec = context_len_sec
        self.frame_rate = vap_process_rate

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

        self.process_time_abs = -1

        self.e1_context = Deque(maxlen=CALC_PROCESS_TIME_INTERVAL - 1)
        self.e1_context.append(np.zeros([1, 1, 256], dtype=np.float32))
        self.e2_context = Deque(maxlen=CALC_PROCESS_TIME_INTERVAL - 1)
        self.e2_context.append(np.zeros([1, 1, 256], dtype=np.float32))

        self.list_process_time_context = []

    def process_vap(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
    ):
        """

        Parameters
        ----------
        x1: np.ndarray
            Left frame. [frame_size]

        x2: np.ndarray
            Right frame. [frame_size]
        """
        x1_ = x1[np.newaxis, np.newaxis, ...]
        x2_ = x2[np.newaxis, np.newaxis, ...]

        p_now, p_future, vad1, vad2, e1, e2 = super().process_vap(
            input_datas=[
                x1_,
                x2_,
                np.concatenate(self.e1_context, axis=1),
                np.concatenate(self.e2_context, axis=1),
            ]
        )
        self.e1_context.append(e1)
        self.e2_context.append(e2)

        self.result_p_now = p_now.squeeze()
        self.result_p_future = p_future.squeeze()


def is_package_installed(package_name: str):
    """Checks if the specified package is installed.

    Parameters
    ----------
    package_name: str
        Name of the package to be checked.

    Returns
    -------
    result: bool
        True if the package is installed, false otherwise.
    """
    return importlib.util.find_spec(package_name) is not None

if __name__ == "__main__":

    # Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vap_onnx_model", type=str, default='./vap_state_dict_jp_20hz_2500msec.onnx')
    parser.add_argument("--filename_output", type=str, default='./output_onnx_offline.txt')
    parser.add_argument("--input_wav_left", type=str, default='../input/wav_sample/jpn_inoue_16k.wav')
    parser.add_argument("--input_wav_right", type=str, default='../input/wav_sample/jpn_sumida_16k.wav')
    parser.add_argument("--vap_process_rate", type=int, default=20)
    parser.add_argument("--context_len_sec", type=float, default=5.0)
    parser.add_argument("--gpu", action='store_true')
    args = parser.parse_args()

    vap_onnx_model: str = args.vap_onnx_model
    filename_output: str = args.filename_output
    input_wav_left: str = args.input_wav_left
    input_wav_right: str = args.input_wav_right
    vap_process_rate: int = args.vap_process_rate
    context_len_sec: float = args.context_len_sec
    gpu: bool = args.gpu

    model_ext: str = os.path.splitext(vap_onnx_model)[1][1:].lower()
    runtime: str = None
    if model_ext == 'onnx':
        if not is_package_installed('onnxruntime'):
            print(Color.RED('ERROR: onnxruntime is not installed. pip install onnxruntime or pip install onnxruntime-gpu'))
            sys.exit(0)
        runtime = 'onnx'
    elif model_ext == 'tflite':
        if is_package_installed('tflite_runtime'):
            runtime = 'tflite_runtime'
        elif is_package_installed('tensorflow'):
            runtime = 'tensorflow'
        else:
            print(Color.RED('ERROR: tflite_runtime or tensorflow is not installed.'))
            print(Color.RED('ERROR: https://github.com/PINTO0309/TensorflowLite-bin'))
            print(Color.RED('ERROR: https://github.com/tensorflow/tensorflow'))
            sys.exit(0)

    if not gpu:
        providers = [
            'CPUExecutionProvider',
        ]
    else:
        providers = [
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]
    print(Color.GREEN('Provider parameters:'))
    pprint(providers)

    vap = \
        VAPRealTime(
            runtime=runtime,
            vap_onnx_model=vap_onnx_model,
            providers=providers,
            vap_process_rate=vap_process_rate,
            context_len_sec=context_len_sec,
        )

    # Load wav
    data_left, _ = sf.read(file=args.input_wav_left, dtype='float32')
    data_right, _ = sf.read(file=args.input_wav_right, dtype='float32')

    # Send data to the vap
    result = []
    frame_size = vap.audio_frame_size
    shift = vap.audio_frame_size - vap.frame_contxt_padding
    total_time = len(data_left) / vap.sampling_rate

    for i in range(0, len(data_left), shift):

        if i + frame_size > len(data_left):
            break

        time_stamp = float(i + frame_size) / vap.sampling_rate

        print('Processing frame: ', time_stamp, ' / ', total_time, end='\r')

        data_left_frame = data_left[i:i+frame_size]
        data_right_frame = data_right[i:i+frame_size]

        vap.process_vap(data_left_frame, data_right_frame)

        p_now = vap.result_p_now
        p_future = vap.result_p_future

        result.append({
            't': time_stamp,
            'p_now': p_now,
            'p_future': p_future
        })

    # Save result
    with open(args.filename_output, 'w') as f:

        # Write header
        f.write('time_sec,p_now(0=left),p_now(1=right),p_future(0=left),p_future(1=right)\n')

        for r in result:
            f.write(str(r['t']) + ',')
            f.write(str(r['p_now'][0]) + ',')
            f.write(str(r['p_now'][1]) + ',')
            f.write(str(r['p_future'][0]) + ',')
            f.write(str(r['p_future'][1]) + '\n')

    print('Generated output file: ', args.filename_output)

