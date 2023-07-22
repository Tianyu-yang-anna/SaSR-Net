from typing import * 

import sys 
import time 
from collections import OrderedDict
import argparse
import os 
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F 

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

sys.path.append("net_grd_avst")

from dataloader_avst import AVQA_dataset, ToTensor

logging.basicConfig(level=logging.DEBUG)

def dataloader_benchmark(dataset: Dataset, max_batch_size: int=64, bs_step: int=1, max_num_workers: int=32, nw_step: int=1, iters: int=20, *argsv, **kwargv):
    benchmarks: OrderedDict = OrderedDict()
    for bs in range(32, max_batch_size + 1, bs_step):
        for nw in range(2, max_num_workers + 1, nw_step):
            logging.debug(f"batch_size = {bs} | num_workers = {nw} | start")
            dataloader: DataLoader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True)
            time_start: float = time.time()
            iter_start_time: float = time.time()
            for i, _ in enumerate(dataloader):
                if i >= iters:
                    break 
                iter_end_time: float = time.time()
                logging.debug(f"batch_size = {bs} | num_workers = {nw} | iter = {i} | iter_time = {iter_end_time - iter_start_time}")
                iter_start_time = time.time()
            time_end: float = time.time()
            benchmarks[(bs, nw)] = (time_end - time_start) / iters
            info: str = f"batch_size = {bs} | num_workers = {nw} | avg_time = {benchmarks[(bs, nw)]} s"
            logging.info(info)
    return benchmarks


def main():
    print(
        dataloader_benchmark(AVQA_dataset(label="./data/json/avqa-train.json", audio_dir="./data/feats/vggish", video_res14x14_dir="./data/feats/res18_14x14",
                                    transform=transforms.Compose([ToTensor()]), mode_flag='train'), bs_step=1, nw_step=2, iters=2)
    ) 


if __name__ == "__main__":
    main()