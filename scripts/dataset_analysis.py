import json
from typing import Any 
from torch.utils.data import Dataset, DataLoader, Sampler
import ast 
import torch
from typing import * 
from collections import Counter, defaultdict


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

        self.samples = json.load(open(label, 'r'))
        self.max_len = 14    # question length
        
        self.ques_vocab = ques_vocab
        self.ans_vocab = ans_vocab
        self.word_to_ix = {word: i for i, word in enumerate(self.ques_vocab)}


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        
        sample = self.samples[idx]
        
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
        # start_time = time.time()
        answer = sample['anser']
        label = ids_to_multinomial(answer, self.ans_vocab)

        return sample, label, answer


items: Set[str] = {'cello', 'congas', 'pipa', 'ukulele', 'piano', 'accordion', 'clarinet', 'guzheng', 'saxophone', 'drum', 'violin', 'bagpipe', 'bassoon', 'acoustic_guitar', 'banjo', 'electric_bass', 'flute', 'trumpet', 'erhu', 'xylophone', 'tuba', 'suona'}

question_content_categories = [
    # label only
    {"What kind of instrument is the <LRer> instrument?", "What kind of musical instrument is it?", "What is the <TH> instrument that comes in?", "Where is the <FL> sounding instrument?", "What is the <LR> instrument of the <FL> sounding instrument?"},
    # template only
    {"How many sounding <Object> in the video？", "Which <Object> makes the sound <FL>?", "Is the <Object> more rhythmic than the <Object>?", "Is the <Object> on the <LR> louder than the <Object> on the <LR>?", "Is the <Object> on the <LR> more rhythmic than the <Object> on the <LR>?", "Is the <Object> louder than the <Object>?", "Is the <Object> playing longer than the <Object>?"},
    # both template and label
    {"What is the instrument on the <LR> of <Object>?", "Which is the musical instrument that sounds at the same time as the <Object>?", "Which instrument makes sounds <BA> the <Object>?"},
    # Special 1
    {"How many <Object> are in the entire video?"},
    # Get template if yes
    {"Is there a <Object> sound?", "Are there <Object> and <Object> instruments in the video?", "Are there <Object> and <Object> sound?"},
    # Undefined cases
    {"Is the <Object> in the video always playing?", "Is there a <Object> in the entire video?"}
]

class FilterSamples:
    def __init__(self) -> None:
        pass
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

def filter_samples(data):
    sample, label, answer = data
    video_id = sample["video_id"]
    question_id = sample["question_id"]
    type_ = sample["type"]
    question_content = sample["question_content"].strip()
    templ_values = eval(sample["templ_values"])
    
    i: int
    res: List[str] = []
    
    for idx in range(len(question_content_categories)):
        if question_content in question_content_categories[idx]:
            i = idx
            
    if i == 0:
        res = [answer]
    
    elif i == 1:
        res = templ_values
    
    elif i == 2:
        res = [answer] + templ_values
        
    elif i == 3:
        if answer != "zero":
            res = templ_values
            
    elif i == 4:
        if answer == "yes":
            res = templ_values
            
    return list(filter(lambda x: x in items, res))
    


def main() -> None:
    dataset: Dataset = AVQA_dataset(label="./data/json/avqa-train.json", audio_dir="./data/feats/vggish", video_res14x14_dir="./data/feats/res18_14x14")
    
    questions: DefaultDict[str, int] = defaultdict(int)
    keywords: DefaultDict[str, int] = defaultdict(int)
    labels: DefaultDict[str, int] = defaultdict(int)
    cnt: int = 0
    labels = {key: 0 for key in items}
    for i, data in enumerate(dataset):
        # sample, label = data
        sample, label, answer = data
        video_id = sample["video_id"]
        question_id = sample["question_id"]
        type_ = sample["type"]
        question_content = sample["question_content"]
        templ_values = sample["templ_values"]
        
        questions[question_content] += 1
        # for templ_value in eval(templ_values):
        #     keywords[templ_value] += 1
        # labels[label] += 1
        
        for item in items:
            if item in question_content or item in templ_values or item in answer:
                cnt += 1
                for x in filter_samples(data):
                    labels[x] += 1
                print(data)
                print(filter_samples(data))
                print() 
    print(labels)
    print(sum(labels.values()))
        
    
    # print(cnt)
    
    # for question, num in questions.items():
    #     print(f"{question} : {num}")
    # print()
    
    # for keyword, num in keywords.items():
    #     print(f"{keyword} : {num}")
    # print()

    # for i, ans in enumerate(dataset.ans_vocab):
    #     print(f"{ans} : {labels[i]}")
    
    # for keyword, num in keywords.items():
    #     if keyword not in dataset.ans_vocab:
    #         print(keyword)

    

if __name__ == "__main__":
    main()
