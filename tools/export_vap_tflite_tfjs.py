import os
import onnx2tf
import argparse
from tensorflowjs.converters import convert_tf_saved_model

if __name__ == "__main__":

    # Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vap_onnx_model", type=str, default='./vap_state_dict_jp_20hz_2500msec_static.onnx')
    args = parser.parse_args()

    # onnx -> tflite/saved_model
    onnx2tf.convert(
        input_onnx_file_path=args.vap_onnx_model,
        output_folder_path="tflite_vap",
        keep_shape_absolutely_input_names=[
            "data_left_frame",
            "data_right_frame",
            "e1_context",
            "e2_context",
        ],
        enable_rnn_unroll=True,
        check_onnx_tf_outputs_elementwise_close_full=True,
        replace_to_pseudo_operators=["Erf"],
    )

    # saved_model -> tfjs
    convert_tf_saved_model(
        saved_model_dir="tflite_vap",
        output_dir="tfjs_vap",
        weight_shard_size_bytes=1024*1024*50,
    )
