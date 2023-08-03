from typing import * 
import sys 

sys.path.append("script")

from audio_tag import AudioTagging

import json
from typing import Any 
from torch.utils.data import Dataset, DataLoader, Sampler
import ast 
import torch
from typing import * 
from collections import Counter, defaultdict

from argparse import Namespace 

args = Namespace(
    sample_rate=32000,
    window_size=1024,
    hop_size = 320,
    mel_bins = 64,
    fmin = 50,
    fmax = 14000,
    classes_num = 527,
)


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
    
    
class GetScoreFromQA:
    def __init__(self):
        self.question_content_categories: List[Set[str]] = [
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
        self.items: Set[str] = {'cello', 'congas', 'pipa', 'ukulele', 'piano', 'accordion', 'clarinet', 'guzheng', 'saxophone', 'drum', 'violin', 'bagpipe', 'bassoon', 'acoustic_guitar', 'banjo', 'electric_bass', 'flute', 'trumpet', 'erhu', 'xylophone', 'tuba', 'suona'}

    def __call__(self, data: Tuple[Any, Any, Any]) -> Dict[str, float]:
        
        sample, label, answer = data
        video_id = sample["video_id"]
        question_id = sample["question_id"]
        type_ = sample["type"]
        question_content = sample["question_content"].strip()
        templ_values = eval(sample["templ_values"])
        
        i: int = -1
        items_set: Set[str] = set()
        
        for idx in range(len(self.question_content_categories)):
            if question_content in self.question_content_categories[idx]:
                i = idx
                
        if i == -1:
            return dict() 
                
        if i == 0:
            items_set.add(answer)
        
        elif i == 1:
            items_set.update(templ_values)
        
        elif i == 2:
            items_set.add(answer)
            items_set.update(templ_values)
            
        elif i == 3:
            if answer != "zero":
                items_set.update(templ_values)
                
        elif i == 4:
            if answer == "yes":
                items_set.update(templ_values)
        
        return {key: 1.0 for key in items_set}
    
def main() -> None:
    items: Set[str] = {'cello', 'congas', 'pipa', 'ukulele', 'piano', 'accordion', 'clarinet', 'guzheng', 'saxophone', 'drum', 'violin', 'bagpipe', 'bassoon', 'acoustic_guitar', 'banjo', 'electric_bass', 'flute', 'trumpet', 'erhu', 'xylophone', 'tuba', 'suona'}
    
    audio_tagging: AudioTagging = AudioTagging(checkpoint_path="pretrained/Cnn14_mAP=0.431.pth", **vars(args))
    dataset: Dataset = AVQA_dataset(label="./data/json/avqa-train.json", audio_dir="./data/feats/vggish", video_res14x14_dir="./data/feats/res18_14x14")
    get_score_from_qa: Callable[[Any], Dict[str, float]] = GetScoreFromQA()
    new_data = list()
    
    for i, data in enumerate(dataset):
        print(f"Current: {i}")
        sample, label, answer = data
        score_from_model: Dict[str, float] = audio_tagging(f"../temp/MUSIC-AVQA/data/audio/{sample['video_id']}.wav")
        score_from_qa: Dict[str, float] = get_score_from_qa(data)
        final_score: Dict[str, float] = {}
        for item in items:
            final_score[item] = str(max(score_from_model[item], score_from_qa.get(item, 0.0)))
        data_updated = sample
        data_updated["items"] = final_score
        new_data.append(data_updated)
    
    # res = json.dumps(new_data)
    
    with open("test.json", "w") as f:
        json.dump(new_data, f, indent=4)
    

        
if __name__ == "__main__":
    main() 