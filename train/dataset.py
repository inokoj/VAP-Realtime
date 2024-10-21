import os
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from glob import glob
from os.path import join
import json
from typing import Dict, Union

from audio import load_waveform
from utils import vad_list_to_onehot

DATASETS = ["fisher", "switchboard", "candor", "callhome", "callfriend", "maptask"]

def load_df(path):
    def _vl(x):
        return json.loads(x)
        # if isinstance(x, str):
        #     if len(x) > 0:
        #         return json.loads(x)
        # return x

    def _session(x):
        return str(x)

    converters = {
        "vad_list": _vl,
        "session": _session,
    }
    return pd.read_csv(path, converters=converters)


def combine_dsets(root):
    splits = {}
    for split in ["train", "val", "test"]:
        tmp = []
        for file in glob(join(root, f"{split}*")):
            tmp.append(load_df(file))
        splits[split] = pd.concat(tmp)

    new_name = basename(root)
    splits["train"].to_csv(f"data/{new_name}_train.csv", index=False)
    splits["val"].to_csv(f"data/{new_name}_val.csv", index=False)
    splits["test"].to_csv(f"data/{new_name}_test.csv", index=False)
    print(f"Saved splits to data/{new_name}_[train/val/test].csv")


class VapDataset(Dataset):
    def __init__(
        self,
        path,
        horizon: float = 2,
        sample_rate: int = 16_000,
        frame_hz: int = 50,
        mono: bool = False,
    ):
        self.path = path
        self.df = load_df(path)

        self.sample_rate = sample_rate
        self.frame_hz = frame_hz
        self.horizon = horizon
        self.mono = mono

        self.audio_path_conv_cash = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Union[Tensor, str, int, float]]:
        d = self.df.iloc[idx]
        # Duration can be 19.99999999999997 for some clips and result in wrong vad-shape
        # so we round it to nearest second
        
        # print(os.uname()[1])
        # print(d["audio_path"])
        audio_path = str(d["audio_path"])

        dur = round(d["end"] - d["start"])
        w, _ = load_waveform(
            audio_path,
            start_time=d["start"],
            end_time=d["end"],
            sample_rate=self.sample_rate,
            mono=self.mono,
        )
        vad = vad_list_to_onehot(
            d["vad_list"], duration=dur + self.horizon, frame_hz=self.frame_hz
        )

        return {
            "session": d["session"],
            "waveform": w,
            "vad": vad,
            "dataset": d["dataset"],
        }


if __name__ == "__main__":
    from os.path import basename

    # root = "data/sliding_window_ad20_ov1_ho2"
    # splits = combine_dsets(root)

    dset = VapDataset(path="data/sliding_window_ad20_ov1_ho2_val.csv")

    d = dset[0]

    dloader = DataLoader(dset, batch_size=4, num_workers=4, shuffle=True)

    batch = next(iter(dloader))
