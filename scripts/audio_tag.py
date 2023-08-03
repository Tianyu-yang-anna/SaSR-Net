from typing import *
import sys
from typing import Any 
import numpy as np 
from pathlib import Path
import os 
from argparse import Namespace
import csv

import torch 
import torch.nn as nn 

import librosa

sys.path.append(".")

import panns 

args = Namespace(
    sample_rate=32000,
    window_size=1024,
    hop_size = 320,
    mel_bins = 64,
    fmin = 50,
    fmax = 14000,
    classes_num = 527,
)

items: Set[str] = {'cello', 'congas', 'pipa', 'ukulele', 'piano', \
    'accordion', 'clarinet', 'guzheng', 'saxophone', 'drum', 'violin', \
        'bagpipe', 'bassoon', 'acoustic_guitar', 'banjo', 'electric_bass', \
            'flute', 'trumpet', 'erhu', 'xylophone', 'tuba', 'suona'}

# Load label
with open('data/test_data/class_labels_indices.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    lines = list(reader)
    
labels = []
ids = []    # Each label has a unique id such as "/m/068hy"
for i1 in range(1, len(lines)):
    id = lines[i1][1]
    label = lines[i1][2]
    ids.append(id)
    labels.append(label)

classes_num = len(labels)


class GetItemScore:
    def __init__(self):
        self.group1 = {
            "cello",
            "accordion",
            "clarinet",
            "saxophone",
            "banjo",
            "flute", 
            "trumpet", 
            "ukulele"
        }
        
        self.group2 = {
            "congas",
            "pipa",
            "guzheng",
            "bassoon",
            "erhu",
            "tuba",
            "suona"
        }
        
        self.group3 = {
            "piano"
        }
         
    def __call__(self, item: str, score_dict: Dict[str, float]):
        if item in self.group1:
            return score_dict[item.capitalize()]
        elif item in self.group2:
            return 0 
        elif item == "piano":
            return max(score_dict["Piano"], score_dict["Electric piano"])
        elif item == "drum":
            return max(score_dict["Drum kit"], score_dict["Drum machine"], score_dict["Drum"], score_dict["Snare drum"], score_dict["Drum roll"], score_dict["Bass drum"], score_dict["Drum and bass"])
        elif item == "violin":
            return score_dict["Violin, fiddle"]
        elif item == "bagpipe":
            return score_dict["Bagpipes"]
        elif item == "acoustic_guitar":
            return score_dict["Steel gAcoustic guitaruitar, slide guitar"]
        elif item == "electric_bass":
            return score_dict["Bass guitar"]
        elif item == "xylophone":
            return score_dict["Marimba, xylophone"]

        
class AudioTagging:
    def __init__(self, checkpoint_path: Union[str, Path], *args, **kwargs):
        self.model: panns.Cnn14_16k = panns.Cnn14_16k(**kwargs)
        checkpoint: Dict[str, Any] = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model"])
        self.device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.args: Namespace = Namespace(**kwargs)
        
        self.model.to(self.device)
        self.model.eval()
        
        self.get_item_score = GetItemScore()
        # pass 
        
    def __call__(self, audio_path: str, *args: Any, **kwds: Any) -> Any:
        res: Dict[str, float] = dict() 
        
        # Load audio
        (waveform, _) = librosa.core.load(audio_path, sr=self.args.sample_rate, mono=True)

        waveform = torch.Tensor(waveform[None, :]).float()    # (1, audio_length)
        waveform = waveform.to(self.device)
        
        with torch.no_grad():
            batch_output_dict: Dict[str, Any] = self.model(waveform, None)
        
        clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()[0]
        """(classes_num,)"""
        
        sorted_indexes = np.argsort(clipwise_output)[::-1]
        
        # # Print audio tagging top probabilities
        # for k in range(10):
        #     print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]], 
        #         clipwise_output[sorted_indexes[k]]))
        
        for k in range(len(sorted_indexes)):
            res[np.array(labels)[sorted_indexes[k]]] = clipwise_output[sorted_indexes[k]]

        # # Print embedding
        # if 'embedding' in batch_output_dict.keys():
        #     embedding = batch_output_dict['embedding'].data.cpu().numpy()[0]
        #     print('embedding: {}'.format(embedding.shape))
        
        ret: Dict[str, str] = {key: self.get_item_score(key, res) for key in items}

        return ret


def main(argc: int, argv: List[str]):
    audio_tagging: AudioTagging = AudioTagging(checkpoint_path="pretrained/Cnn14_mAP=0.431.pth", **vars(args))
    pred: Dict[str, float] = audio_tagging("../temp/MUSIC-AVQA/data/audio/00001000.wav")
    print(pred)
    # checkpoint_path: Path = Path("")
    # state_dict = torch.load("pretrained/Cnn14_mAP=0.431.pth")
    # print(state_dict["model"].keys())

if __name__ == "__main__":
    main(len(sys.argv), sys.argv) 