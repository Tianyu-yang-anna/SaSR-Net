import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import argparse
import time
import sys

# Question
class QstEncoder(nn.Module):

    def __init__(self, qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size):

        super(QstEncoder, self).__init__()
        self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        # 2 for hidden and cell states
        self.fc = nn.Linear(2 * num_layers*hidden_size, embed_size)

    def forward(self, question):

        # [batch_size, max_qst_length=30, word_embed_size=300]
        qst_vec = self.word2vec(question)
        qst_vec = self.tanh(qst_vec)
        # [max_qst_length=30, batch_size, word_embed_size=300]
        qst_vec = qst_vec.transpose(0, 1)
        self.lstm.flatten_parameters()
        # [num_layers=2, batch_size, hidden_size=512]
        _, (hidden, cell) = self.lstm(qst_vec)
        # [num_layers=2, batch_size, 2*hidden_size=1024]
        qst_feature = torch.cat((hidden, cell), 2)
        # [batch_size, num_layers=2, 2*hidden_size=1024]
        qst_feature = qst_feature.transpose(0, 1)
        # [batch_size, 2*num_layers*hidden_size=2048]
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)
        qst_feature = self.tanh(qst_feature)
        # [batch_size, embed_size]
        qst_feature = self.fc(qst_feature)

        return qst_feature
