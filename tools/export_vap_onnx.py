import torch
import soundfile as sf
from tools.vap_static import VAPRealTimeStatic
import argparse
import os
import onnx
from onnxsim import simplify

if __name__ == "__main__":

    # Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vap_model", type=str, default='../asset/vap/vap_state_dict_jp_20hz_2500msec.pt')
    parser.add_argument("--cpc_model", type=str, default='../asset/cpc/60k_epoch4-d0f474de.pt')
    parser.add_argument("--vap_process_rate", type=int, default=20)
    parser.add_argument("--context_len_sec", type=float, default=5.0)
    parser.add_argument("--cpc_encoder_feature_length", type=int, default=256)
    parser.add_argument("--onnx_opset", type=int, default=17)
    args = parser.parse_args()

    device = torch.device('cpu')
    print('Device: ', device)

    vap = VAPRealTimeStatic(
        vap_model=args.vap_model,
        cpc_model=args.cpc_model,
        device=device,
        frame_rate=args.vap_process_rate,
        context_len_sec=args.context_len_sec,
    )

    frame_size = vap.audio_frame_size

    # vap_state_dict_jp_20hz_2500msec.pt
    #   data_left_frame.shape[x1]: (1120,), data_right_frame.shape[x2]: (1120,)
    #   x1_.shape: torch.Size([1, 1, 1120]), x2_.shape: torch.Size([1, 1, 1120])
    data_left_frame = torch.randn([1, 1, frame_size])
    data_right_frame = torch.randn([1, 1, frame_size])
    e1_context = torch.randn(1, 1, args.cpc_encoder_feature_length)
    e2_context = torch.randn(1, 1, args.cpc_encoder_feature_length)

    vap_model_without_ext = os.path.splitext(os.path.basename(args.vap_model))[0]
    onnx_file = f"{vap_model_without_ext}.onnx"

    # onnx export
    torch.onnx.export(
        model=vap,
        args=(data_left_frame, data_right_frame, e1_context, e2_context),
        f=onnx_file,
        opset_version=args.onnx_opset,
        input_names=['data_left_frame', 'data_right_frame', 'e1_context', 'e2_context'],
        output_names=['p_now', 'p_future', 'vad1', 'vad2', 'e1', 'e2'],
        dynamic_axes={
            'p_now' : {0: '1'},
            'p_future' : {0: '1'},
            'vad1' : {0: '1'},
            'vad2' : {0: '1'},
            'e1_context' : {1: 'M'},
            'e2_context' : {1: 'M'},
        },
    )

    # onnx shape inference
    original_onnx = onnx.load(onnx_file)
    shape_infered_onnx = onnx.shape_inference.infer_shapes(original_onnx)
    onnx.save(shape_infered_onnx, onnx_file)

    # onnx simplify
    shape_infered_onnx = onnx.load(onnx_file)
    simplified_onnx, check = simplify(shape_infered_onnx)
    onnx.save(simplified_onnx, onnx_file)
