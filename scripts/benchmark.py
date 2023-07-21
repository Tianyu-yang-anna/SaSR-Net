from typing import * 

import sys 
import time 
from collections import OrderedDict
import argparse
import os 

import torch
import torch.nn as nn
import torch.nn.functional as F 

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

sys.path.append("net_grd_avst")

from dataloader_avst import AVQA_dataset, ToTensor

def dataloader_benchmark(dataset: Dataset, max_batch_size: int=64, bs_step: int=1, max_num_workers: int=16, iters: int=20, *argsv, **kwargv):
    benchmarks: OrderedDict = OrderedDict()
    for bs in range(1, max_batch_size, bs_step):
        for nw in range(max_num_workers):
            dataloader: DataLoader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True)
            time_start: int = time.time()
            for i, _ in enumerate(dataloader):
                if i >= iters:
                    break 
            time_end: int = time.time()
            benchmarks[(bs, nw)] = (time_end - time_start) / iters
    return benchmarks


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Audio-Visual Question Answering')

    parser.add_argument(
        "--audio_dir", type=str, default="./data/feats/vggish", help="audio dir")
    # parser.add_argument(
    #     "--video_dir", type=str, default='/home/guangyao_li/dataset/avqa/avqa-frames-1fps', help="video dir")
    parser.add_argument(
        "--video_res14x14_dir", type=str, default="./data/feats/r2plus1d_18/r2plus1d_18", help="res14x14 dir")
    
    parser.add_argument(
        "--label_train", type=str, default="./data/json/avqa-train.json", help="train csv file")
    parser.add_argument(
        "--label_val", type=str, default="./data/json/avqa-val.json", help="val csv file")
    parser.add_argument(
        "--label_test", type=str, default="./data/json/avqa-test.json", help="test csv file")
    parser.add_argument(
        '--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 16)')
    parser.add_argument(
        '--epochs', type=int, default=80, metavar='N', help='number of epochs to train (default: 60)')
    parser.add_argument(
        '--lr', type=float, default=3e-4, metavar='LR', help='learning rate (default: 3e-4)')
    parser.add_argument(
        "--model", type=str, default='AVQA_Fusion_Net', help="with model to use")
    parser.add_argument(
        "--mode", type=str, default='train', help="with mode to use")
    parser.add_argument(
        '--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval', type=int, default=50, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument(
        "--model_save_dir", type=str, default='net_grd_avst/avst_models/', help="model save dir")
    parser.add_argument(
        "--checkpoint", type=str, default='avst', help="save model name")
    parser.add_argument(
        '--gpu', type=str, default='0, 1', help='gpu device number')


    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    torch.manual_seed(args.seed)
    
    print(
        dataloader_benchmark(AVQA_dataset(label=args.label_train, audio_dir=args.audio_dir, video_res14x14_dir=args.video_res14x14_dir,
                                    transform=transforms.Compose([ToTensor()]), mode_flag='train'))
    ) 


if __name__ == "__main__":
    main()