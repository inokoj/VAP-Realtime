<h1>
<p align="center">
Training VAP model
</p>
</h1>
<!-- <p align="center">
README: <a href="README.md">English </a> | <a href="README_JP.md">Japanese (日本語) </a>
</p> -->

This sub repository provides the training code for the VAP model, designed to operate in real-time processing.
We highly recommend using GPUs for optimal performance during training.

Please note that this repository and README are currently under development and will be updated in the near future.

## (1) Install

```bash
$ pip install -r requirements-training.txt
```

## (2) Preparation

To begin, you need to create CSV files that define the training, validation, and test data.
These files should include the following information: the path to the audio file, the start and end times of each segment, VAD data, session ID, and dataset name.

A sample format is shown below:

```csv
audio_path	start	end	vad_list	session	dataset
sample/jpn_mix_16k.wav	0	20	[[[3.77, 4.51], [5.25, 7.84], [8.46, 14.1], [16.73, 17.39], [17.72, 20.33]], [[13.16, 13.4], [13.98, 14.13], [14.56, 17.74], [20.73, 22.0]]]	0	sample
sample/jpn_mix_16k.wav	20	40	[[[3.77, 4.51], [5.25, 7.84], [8.46, 14.1], [16.73, 17.39], [17.72, 20.33]], [[13.16, 13.4], [13.98, 14.13], [14.56, 17.74], [20.73, 22.0]]]	0	sample
sample/jpn_mix_16k.wav	40	60	[[[3.77, 4.51], [5.25, 7.84], [8.46, 14.1], [16.73, 17.39], [17.72, 20.33]], [[13.16, 13.4], [13.98, 14.13], [14.56, 17.74], [20.73, 22.0]]]	0	sample
```

### Audio File Path (1st column)

The audio data should be stereo (2 channels), 16 kHz, and 16-bit format.
Each channel should correspond to a different dialogue participant.
Each audio file should cover the entire dialogue session, so segmentation into smaller clips is not required.

### Audio Segment (2nd and 3rd columns)

The VAP model processes 20-second audio segments in the training.
The input audio should encompass the entire dialogue session as mentioned above, and each row in the CSV file should represent a 20-second segment, with the appropriate start and end times. 
Generally, these segments are continuous, such as 0-20 seconds, 20-40 seconds, etc.

### VAD data (4th column)

For the training process, you will need to provide VAD (Voice Activity Detection) data as reference data.
The format should be a string representing a list, where the top-level list contains two sublists, one for each dialogue participant's VAD data.
Each sublist contains the utterance segments (VAD data), with each segment defined by its relative start and end times within the 20-second window.

Additionally, the VAD data should include 2 seconds beyond the 20-second segment to account for future voice activity, resulting in a total of 22 seconds of data.

### Others (5th and 6th columns)

The session ID and dataset name are arbitrary, so any values are acceptable as long as they are not empty.

## (3) Running the Training Process

To run the training, specify the training and validation data files, along with other parameters, as shown below:

```bash
$ python train.py ^
    --data_train_path sample/train.csv ^
    --data_val_path sample/valid.csv ^
    --data_test_path sample/test.csv ^
    --vap_encoder_type cpc ^
    --vap_cpc_model_pt ../asset/cpc/60k_epoch4-d0f474de.pt ^
    --vap_freeze_encoder 1 ^
    --vap_channel_layers 1 ^
    --vap_cross_layers 3 ^
    --vap_context_limit -1 ^
    --vap_context_limit_cpc_sec -1 ^
    --vap_frame_hz 50 ^
    --event_frame_hz 50 ^
    --opt_early_stopping 0 ^
    --opt_save_top_k 5 ^
    --opt_max_epochs 25 ^
    --opt_saved_dir ./trained_data/ ^
    --data_batch_size 8 ^
    --devices 0 ^
    --seed 0
```

## (4) Evaluation

To run the testing, specify the test data file, along with other parameters, as shown below:

```bash
python evaluation.py ^
    --data_test_path sample/test.csv ^
    --vap_encoder_type cpc ^
    --vap_cpc_model_pt ../asset/cpc/60k_epoch4-d0f474de.pt ^
    --vap_freeze_encoder 1 ^
    --vap_channel_layers 1 ^
    --vap_cross_layers 3 ^
    --vap_context_limit -1 ^
    --vap_context_limit_cpc_sec -1 ^
    --vap_frame_hz 50 ^
    --event_frame_hz 50 ^
    --data_batch_size 8 ^
    --event_equal_hold_shift 0 ^
    --devices 0 ^
    --checkpoint ./trained_data/
```

With the current program, after the testing, you would be able to check the test loss in the file `runs_evaluation/score.csv`.
