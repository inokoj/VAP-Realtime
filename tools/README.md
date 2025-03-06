## 1. Environment
```bash
pip install \
onnx==1.17.0 \
onnxruntime==1.18.1 \
onnxsim==0.4.30 \
soundfile==0.12.1 \
spo4onn==1.0.5
```

## 2. Export ONNX
- Command prompt
```shell
python export_vap_onnx.py ^
--vap_model ../asset/vap/vap_state_dict_jp_20hz_2500msec.pt ^
--cpc_model ../asset/cpc/60k_epoch4-d0f474de.pt ^
--vap_process_rate 20 ^
--context_len_sec 5.0 ^
--cpc_encoder_feature_length 256
```
- PowerShell
```powershell
python export_vap_onnx.py `
--vap_model ../asset/vap/vap_state_dict_jp_20hz_2500msec.pt `
--cpc_model ../asset/cpc/60k_epoch4-d0f474de.pt `
--vap_process_rate 20 `
--context_len_sec 5.0 `
--cpc_encoder_feature_length 256
```
- Bash
```bash
python export_vap_onnx.py \
--vap_model ../asset/vap/vap_state_dict_jp_20hz_2500msec.pt \
--cpc_model ../asset/cpc/60k_epoch4-d0f474de.pt \
--vap_process_rate 20 \
--context_len_sec 5.0 \
--cpc_encoder_feature_length 256
```

## 3. ONNX offline test
- Command prompt
```shell
python vap_offline_onnx.py ^
--vap_onnx_model vap_state_dict_jp_20hz_2500msec_static.onnx ^
--input_wav_left ../input/wav_sample/jpn_inoue_16k.wav ^
--input_wav_right ../input/wav_sample/jpn_sumida_16k.wav ^
--filename_output ./output_offline.txt
```
- PowerShell
```powershell
python vap_offline_onnx.py `
--vap_onnx_model vap_state_dict_jp_20hz_2500msec_static.onnx `
--input_wav_left ../input/wav_sample/jpn_inoue_16k.wav `
--input_wav_right ../input/wav_sample/jpn_sumida_16k.wav `
--filename_output ./output_offline.txt
```
- Bash
```bash
python vap_offline_onnx.py \
--vap_onnx_model vap_state_dict_jp_20hz_2500msec_static.onnx \
--input_wav_left ../input/wav_sample/jpn_inoue_16k.wav \
--input_wav_right ../input/wav_sample/jpn_sumida_16k.wav \
--filename_output ./output_offline.txt
```

## 4. Visualize
- Command prompt
```shell
python ../output/offline_prediction_visualizer/main.py ^
--left_audio ../input/wav_sample/jpn_inoue_16k.wav ^
--right_audio ../input/wav_sample/jpn_sumida_16k.wav ^
--prediction ./output_offline.txt
```
- PowerShell
```powershell
python ../output/offline_prediction_visualizer/main.py `
--left_audio ../input/wav_sample/jpn_inoue_16k.wav `
--right_audio ../input/wav_sample/jpn_sumida_16k.wav `
--prediction ./output_offline.txt
```
- Bash
```bash
python ../output/offline_prediction_visualizer/main.py \
--left_audio ../input/wav_sample/jpn_inoue_16k.wav \
--right_audio ../input/wav_sample/jpn_sumida_16k.wav \
--prediction ./output_offline.txt
```

## 5. tflte/tfjs conversion
```bash
pip install \
tensorflow \
tensorflowjs \
tensorflow_decision_forests \
ydf \
tensorflow_hub \
onnx2tf>=1.26.8
```
```bash
python export_vap_tflite_tfjs.py \
--vap_onnx_model ./vap_state_dict_jp_20hz_2500msec_static.onnx
```

## 6. tfjs offline test
Starting the HTTP Server.
```bash
python -m http.server 8088
```
Start your browser and access the following URL.
```
http://127.0.0.1:8088/vap_offline_tfjs.html
```
- Results
![Image](https://github.com/user-attachments/assets/b1e02b93-9c9c-496b-807e-0f1ba3de8c52)