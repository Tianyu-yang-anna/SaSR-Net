from typing import Optional, Sequence
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.dataloader import _BaseDataLoaderIter, _collate_fn_t, _worker_init_fn_t
from torchvision import transforms, utils
import pandas as pd
import ast
import json
from PIL import Image
from munch import munchify
import time
import random
import torch.nn.functional as F


# def TransformImage(img):

#     transform_list = []
#     mean = [0.43216, 0.394666, 0.37645]
#     std = [0.22803, 0.22145, 0.216989]

#     transform_list.append(transforms.Resize([224,224]))
#     transform_list.append(transforms.ToTensor())
#     transform_list.append(transforms.Normalize(mean, std))
#     trans = transforms.Compose(transform_list)
#     frame_tensor = trans(img)
    
#     return frame_tensor


# def load_frame_info(img_path):

#     img = Image.open(img_path).convert('RGB')
#     frame_tensor = TransformImage(img)

#     return frame_tensor


# def image_info(video_name):

#     path = "/home/guangyao_li/dataset/avqa/avqa-frames-8fps"
#     img_path = os.path.join(path, video_name)

#     img_list = os.listdir(img_path)
#     img_list.sort()

#     select_img = []
#     for frame_idx in range(0,len(img_list),8):
#         if frame_idx < 475:
#             video_frames_path = os.path.join(img_path, str(frame_idx+1).zfill(6)+".jpg")

#             frame_tensor_info = load_frame_info(video_frames_path)
#             select_img.append(frame_tensor_info.cpu().numpy())

#     select_img = np.array(select_img)

#     return select_img


def ids_to_multinomial(id, categories):
    """ label encoding
    Returns:
      1d array, multimonial representation, e.g. [1,0,1,0,0,...]
    """
    id_to_idx = {id: index for index, id in enumerate(categories)}

    return id_to_idx[id]

class AVQA_dataset(Dataset):

    def __init__(self, label, audio_dir, video_res14x14_dir, transform=None, mode_flag='train'):


        samples = json.load(open('./data/json/avqa-train.json', 'r'))

        # nax =  nne
        ques_vocab = ['<pad>']
        ans_vocab = []
        i = 0
        for sample in samples:
            i += 1
            question = sample['question_content'].rstrip().split(' ')
            question[-1] = question[-1][:-1]

            p = 0
            for pos in range(len(question)):
                if '<' in question[pos]:
                    question[pos] = ast.literal_eval(sample['templ_values'])[p]
                    p += 1

            for wd in question:
                if wd not in ques_vocab:
                    ques_vocab.append(wd)
            if sample['anser'] not in ans_vocab:
                ans_vocab.append(sample['anser'])

        self.ques_vocab = ques_vocab
        self.ans_vocab = ans_vocab
        self.word_to_ix = {word: i for i, word in enumerate(self.ques_vocab)}

        self.samples = json.load(open(label, 'r'))
        self.max_len = 14    # question length

        self.audio_dir = audio_dir
        self.video_res14x14_dir = video_res14x14_dir
        self.transform = transform

        video_list = []
        for sample in self.samples:
            video_name = sample['video_id']
            if video_name not in video_list:
                video_list.append(video_name)

        self.video_list = video_list
        self.video_len = 60 * len(video_list)

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        
        #start_time = time.time()
        sample = self.samples[idx]
        name = sample['video_id']
        audio = np.load(os.path.join(self.audio_dir, name + '.npy'))
        audio = audio[::6, :]
        #end_time = time.time()
        #print("load_audio", end_time - start_time)

        #start_time = time.time()
        # visual_out_res18_path = '/home/guangyao_li/dataset/avqa-features/visual_14x14'
        visual_posi = np.load(os.path.join(self.video_res14x14_dir, name + '.npy'))  
        #end_time = time.time()
        #print("load_vidual_res", end_time - start_time)

        #start_time = time.time()
        # visual_posi [60, 512, 14, 14], select 10 frames from one video
        visual_posi = visual_posi[::6, :]
        video_idx=self.video_list.index(name)
        #end_time = time.time()
        #print("load_vidual_posi", end_time - start_time)

        load_vidual_neg_1, load_vidual_neg_2, load_vidual_neg_3, load_vidual_neg_4, load_vidual_neg_5, load_vidual_neg_6\
            = 0, 0, 0, 0, 0, 0
        #start_time = time.time()
        for i in range(visual_posi.shape[0]):
            #start_time = time.time()
            while(1):
                neg_frame_id = random.randint(0, self.video_len - 1)
                if (int(neg_frame_id/60) != video_idx):
                    break
            #end_time = time.time()
            #load_vidual_neg_1 += end_time - start_time

            neg_video_id = int(neg_frame_id / 60)
            neg_frame_flag = neg_frame_id % 60
            start_time = time.time()
            neg_video_name = self.video_list[neg_video_id]
            #end_time = time.time()
            #load_vidual_neg_2 += end_time - start_time

            #start_time = time.time()
            visual_nega_out_res18=np.load(os.path.join(self.video_res14x14_dir, neg_video_name + '.npy'))
            #end_time = time.time()
            #load_vidual_neg_3 += end_time - start_time

            #start_time = time.time()
            visual_nega_out_res18 = torch.from_numpy(visual_nega_out_res18)
            #end_time = time.time()
            #load_vidual_neg_4 += end_time - start_time

            #start_time = time.time()
            visual_nega_clip=visual_nega_out_res18[neg_frame_flag,:,:,:].unsqueeze(0)
            #end_time = time.time()
            #load_vidual_neg_5 += end_time - start_time

            if(i==0):
                visual_nega=visual_nega_clip
            else:
                #start_time = time.time()
                visual_nega=torch.cat((visual_nega,visual_nega_clip),dim=0)
                #end_time = time.time()
                #load_vidual_neg_6 += end_time - start_time

        #end_time = time.time()
        # print("load_vidual_neg", end_time - start_time)
        # print("load_visual_neg_1", load_vidual_neg_1)
        # print("load_visual_neg_2", load_vidual_neg_2)
        # print("load_visual_neg_3", load_vidual_neg_3)
        # print("load_visual_neg_4", load_vidual_neg_4)
        # print("load_visual_neg_5", load_vidual_neg_5)
        # print("load_visual_neg_6", load_vidual_neg_6)


        # visual nega [60, 512, 14, 14]

        # question
        #start_time = time.time()
        question_id = sample['question_id']
        question = sample['question_content'].rstrip().split(' ')
        question[-1] = question[-1][:-1]

        p = 0
        for pos in range(len(question)):
            if '<' in question[pos]:
                question[pos] = ast.literal_eval(sample['templ_values'])[p]
                p += 1
        if len(question) < self.max_len:
            n = self.max_len - len(question)
            for i in range(n):
                question.append('<pad>')
        idxs = [self.word_to_ix[w] for w in question]
        ques = torch.tensor(idxs, dtype=torch.long)
        #end_time = time.time()
        #print("load_question", end_time - start_time)

        # answer
        #start_time = time.time()
        answer = sample['anser']
        label = ids_to_multinomial(answer, self.ans_vocab)
        label = torch.from_numpy(np.array(label)).long()

        sample = {'audio': audio, 'visual_posi': visual_posi, 'visual_nega': visual_nega, 'question': ques, 'label': label}
        # end_time = time.time()
        # print("load_vidual_anser", end_time - start_time)
        
        if self.transform:
            sample = self.transform(sample)

        return sample
        

class ToTensor(object):

    def __call__(self, sample):

        audio = sample['audio']
        visual_posi = sample['visual_posi']
        visual_nega = sample['visual_nega']
        label = sample['label']
        # label = F.one_hot(sample['label'], num_classes=42)

        return { 
                'audio': torch.from_numpy(audio), 
                'visual_posi': sample['visual_posi'],
                'visual_nega': sample['visual_nega'],
                'question': sample['question'],
                'label': label}
    

class MasterSlaveDataLoader(DataLoader):
    def __iter__(self) -> _BaseDataLoaderIter:
        num_gpus: int = torch.cuda.device_count()
        gpu_id: int = torch.cuda.current_device()
        data_iter: iter = iter(self.dataset)
        num_batches: int = len(self) // num_gpus 
        for i in range(num_batches):
            batch = list()
            for _ in range(num_gpus):
                if gpu_id < num_gpus:
                    try:
                        data = next(data_iter)
                        batch.append(data)
                    except StopIteration:
                        break 
                else:
                    break 
            if batch:
                ret = {}
                values = []
                for key in batch[0].keys():
                    values.clear()
                    for e in batch:
                        values.append(e[key])
                    if (isinstance(values[0], torch.Tensor)):
                        ret[key] = torch.stack(values, dim=0).cuda(gpu_id)
                    else:
                        ret[key] = torch.tensor(np.stack(values, axis=0)).cuda(gpu_id)
                yield ret 