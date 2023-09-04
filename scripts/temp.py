from typing import * 
from pathlib import Path 
import sys 
import json 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import numpy as np 

from torch.utils.data import Dataset, DataLoader 


class Config:
    VIDEO_LIST_FILE_PATH: str = ""
    
    
class Utils:
    @staticmethod 
    def get_video_frame_sample():
        pass 
    
    @staticmethod 
    def get_pretrained_model(model_path: Union[str, Path]) -> nn.Module:
        model: nn.Module = torch.load(model_path)
        return model 
    
    @staticmethod 
    def load_dataloader(dataset_path) -> Dataset:
        dataset: Dataset = MyDataset()
        return Dataset()

    
class MyDataset(Dataset):

    def __init__(self, label, audio_dir, video_res14x14_dir, transform=None, mode_flag='train'):

        self.ques_vocab, self.ans_vocab = self.load_vocabs()
        
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
    
    def load_vocabs(self, sample_path: str = './data/json/avqa-train.json'):
        
        with open('./data/json/avqa-train.json', 'r') as f:
            samples = json.load(f)
            
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
        return ques_vocab, ans_vocab
        

def main():
    pass 

if __name__  == "__main__":
    main() 