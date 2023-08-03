from typing import Optional, Sequence, List, Any, Callable, Dict
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

from typing import *
from collections import OrderedDict

T = TypeVar('T')
S = TypeVar('S')

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


class LRU_cache(Generic[T, S]):
    def __init__(self, max_size: Optional[int]=None) -> None:
        self.cache: OrderedDict[T, S] = OrderedDict()
        self.count: int = 0
        self.max_size: int = max_size
    
    def setdefault(self, key: T, _default_value: Optional[S]=None) -> S:
        if key in self.cache:
            return self.cache[key]
        self.cache[key] = _default_value
        self.count += 1
        if self.max_size is not None and self.count > self.max_size:
            self.cache.popitem(last=False)
        return _default_value
        
    def __len__(self) -> int:
        return self.count


def func_ids_to_multinomial(categories):
    id_to_idx = {id: index for index, id in enumerate(categories)}
    
    def ids_to_multinomial(id):
        """ label encoding
        Returns:
        1d array, multimonial representation, e.g. [1,0,1,0,0,...]
        """

        return id_to_idx[id]
    return ids_to_multinomial

class AVQA_dataset(Dataset):

    def __init__(self, label, audio_dir, video_res14x14_dir, transform=None, mode_flag='train'):
  
        self.train: bool = mode_flag == "train"
        
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
        self.frame_ids: np.ndarray[int] = np.arange(self.video_len)
        
        self.audio_data: LRU_cache[str, np.ndarray[Any]] = LRU_cache(max_size=None)
        self.visual_data: LRU_cache[str, np.ndarray[Any]] = LRU_cache(max_size=None)
        
        self.ids_to_multinomial = func_ids_to_multinomial(self.ans_vocab)
        self.items: List[str] = ['cello', 'congas', 'pipa', 'ukulele', 'piano', 'accordion', 'clarinet', 'guzheng', 'saxophone', 'drum', 'violin', 'bagpipe', 'bassoon', 'acoustic_guitar', 'banjo', 'electric_bass', 'flute', 'trumpet', 'erhu', 'xylophone', 'tuba', 'suona']


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # start_time_ = time.time()
        sample = self.samples[idx]
        name = sample['video_id']
        
        if self.train:
            items: Dict[str, str] = sample["items"]
        else:
            items = {}
            
        # audio = np.load(os.path.join(self.audio_dir, name + '.npy'), mmap_mode='r')
        audio = self.audio_data.setdefault(name, np.load(os.path.join(self.audio_dir, name + '.npy'), mmap_mode='r'))
        audio = audio[::6, :]

        # visual_out_res18_path = '/home/guangyao_li/dataset/avqa-features/visual_14x14'
        # visual_posi = np.load(os.path.join(self.video_res14x14_dir, name + '.npy'))  
        visual_posi = self.visual_data.setdefault(name, np.load(os.path.join(self.video_res14x14_dir, name + '.npy'), mmap_mode='r'))

        # visual_posi [60, 512, 14, 14], select 10 frames from one video
        visual_posi = visual_posi[::6, :]
        video_idx = self.video_list.index(name)
        # valid_frame_ids: np.ndarray[int] = self.frame_ids[self.frame_ids // 60 != video_idx]
        # neg_frame_ids: np.ndarray[int] = np.random.choice(valid_frame_ids, size=visual_posi.shape[0], replace=False)
        neg_frame_ids: List[int] = [random_int(0, self.video_len - 1, lambda x: x // 60 != video_idx) for _ in range(visual_posi.shape[0])]

        #start_time = time.time()
        
        visual_nega_list: List[np.ndarray[int]] = []
        
        for i in range(visual_posi.shape[0]):
            neg_frame_id: int = neg_frame_ids[i]
            
            neg_video_id: int = neg_frame_id // 60
            neg_frame_flag: int = neg_frame_id % 60

            neg_video_name: str = self.video_list[neg_video_id]

            # visual_nega_out_res18: np.ndarray[Any] = np.load(os.path.join(self.video_res14x14_dir, neg_video_name + '.npy'), mmap_mode='r')
            visual_nega_out_res18 = self.visual_data.setdefault(neg_video_name, np.load(os.path.join(self.video_res14x14_dir, neg_video_name + '.npy'), mmap_mode='r'))
            visual_nega_list.append(visual_nega_out_res18[neg_frame_flag,:,:,:])
        
        visual_nega: Any = np.stack(visual_nega_list, axis=0)
        visual_nega: Any = torch.from_numpy(visual_nega)
        # for i in range(visual_posi.shape[0]):
        #     while(1):
        #         neg_frame_id = random.randint(0, self.video_len - 1)
        #         if (int(neg_frame_id/60) != video_idx):
        #             break

        #     neg_video_id = int(neg_frame_id / 60)
        #     neg_frame_flag = neg_frame_id % 60

        #     neg_video_name = self.video_list[neg_video_id]

        #     visual_nega_out_res18=np.load(os.path.join(self.video_res14x14_dir, neg_video_name + '.npy'))
        #     visual_nega_out_res18 = torch.from_numpy(visual_nega_out_res18)

        #     visual_nega_clip=visual_nega_out_res18[neg_frame_flag,:,:,:].unsqueeze(0)

        #     if(i==0):
        #         visual_nega=visual_nega_clip
        #     else:
        #         visual_nega=torch.cat((visual_nega,visual_nega_clip),dim=0)

        
        # end_time = time.time()
        # print(f"{idx}: process video time: {end_time - start_time}")
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

        # answer
        #start_time = time.time()
        answer = sample['anser']
        label = self.ids_to_multinomial(answer)
        label = torch.from_numpy(np.array(label)).long()

        sample = {'audio': audio, 'visual_posi': visual_posi, 'visual_nega': visual_nega, 'question': ques, 'label': label, 'items': self.items_to_embed(items)}
        # end_time = time.time()
        # print("load_vidual_anser", end_time - start_time)
        
        if self.transform:
            sample = self.transform(sample)
            
        # end_time_ = time.time()
        # print(f"{idx}: process all time: {end_time_ - start_time_}")

        return sample
    
    def items_to_embed(self, items: Dict[str, str]) -> np.ndarray:
        res: np.ndarray = np.zeros(len(self.items))
        for i, item in enumerate(self.items):
            res[i] = items.get(item, 0)
        return res 
            
             
    

def random_int(min_value: int=0, max_value: int=10000, filter_key: Callable[[int], bool]=lambda _: True) -> int:
    while True:
        i = random.randint(0, max_value)
        if filter_key(i):
            return i
        

class ToTensor(object):

    def __call__(self, sample):

        audio = sample['audio']
        visual_posi = sample['visual_posi']
        visual_nega = sample['visual_nega']
        label = sample['label']
        items = sample["items"]
        # label = F.one_hot(sample['label'], num_classes=42)

        return { 
            'audio': torch.from_numpy(audio), 
            'visual_posi': sample['visual_posi'],
            'visual_nega': sample['visual_nega'],
            'question': sample['question'],
            'label': label,
            "items": torch.from_numpy(items)}