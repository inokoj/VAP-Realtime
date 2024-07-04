import torch

import soundfile as sf

from vap_main import VAPRealTime
import argparse

from os import environ
environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(mode=True)

torch.manual_seed(0)

if __name__ == "__main__":
    
    # Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vap_model", type=str, default='../asset/vap/vap_state_dict_20hz_jpn.pt')
    parser.add_argument("--cpc_model", type=str, default='../asset/cpc/60k_epoch4-d0f474de.pt')
    parser.add_argument("--filename_output", type=str, default='output_offline.txt')
    parser.add_argument("--input_wav_left", type=str, default='../input/wav_sample/jpn_inoue_16k.wav')
    parser.add_argument("--input_wav_right", type=str, default='../input/wav_sample/jpn_sumida_16k.wav')
    args = parser.parse_args()

    vap = VAPRealTime(args.vap_model, args.cpc_model)
    
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
        
        
        p_future = vap.result_p_future
        p_now = vap.result_p_now
        
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
    
   