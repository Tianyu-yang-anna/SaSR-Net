# AUDIO-VISUAL QUESTION ANSWERING WITH SEMANTIC GUIDANCE

Implementation for our paper:

**Learning to Answer Questions in Dynamic Audio-Visual Scenarios**

[Tianyu Yang](https://github.com/tianyu-yang1998), [Yiyang Nan](www.google.com), [Yapeng Tian](https://yapengtian.org/), [Xiangliang Zhang](www.google.com)

## Usage

### Set up Environment

Clone this repo
```
git clone https://github.com/tianyu-yang1998/SaSR-Net
```

Install Conda virtual environment

```
conda create --name sasr-net python=3.8
```

Activate Conda virtual environment

```
source activate sasr-net
```

Install all the dependencies

```
pip install -r requirements.txt
```

### Download Dataset

For dataset download, please check on this repo [Github: MUSIC-AVQA](https://github.com/GeWu-Lab/MUSIC-AVQA#whats-audio-visual-question-answering-task). The dataset setup will follow the **Usage/Download data** section in the README file.

For json files, please use the ones contained in this repo. Since we have updated semantic labels on them.

## Dataset Processing

For dataset processing, please also check on this repo [Github: MUSIC-AVQA](https://github.com/GeWu-Lab/MUSIC-AVQA#whats-audio-visual-question-answering-task). The dataset setup will follow the **Usage/Data pre-processing** section and **Usage/Feature extraction** section in the README file.

### Download Pretrained Models

There are two kinds of pretrained models.

1. Feature extraction backbone models, which should have been downloaded and put to the right place in Section **Dataset Processing**.
2. Grounding gen model, which could be used as part of pre-trained weights of our SaSR-Net. It's already inside `./pretrained`.

### Train

```
python sasr_net/main.py --mode train --audio_dir "./data/feats/vggish" --video_res14x14_dir "./data/feats/res18_14x14"
```

### Test

```
python sasr_net/main.py --mode test --audio_dir "./data/feats/vggish" --video_res14x14_dir "./data/feats/res18_14x14" --checkpoint ${CHECKPOINT_NAME}
```
