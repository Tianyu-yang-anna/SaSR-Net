import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import argparse
import time
import sys

sys.path.append("./sasr_net")
sys.path.append("./sasr_net/modules")

from modules.sasr import SourceAwareSemanticRepresentation
from modules.visual_net import resnet18
from modules.qst_net import QstEncoder


class SaSR_Net(nn.Module):

    def __init__(self, dim_audio: int = 128, dim_visual: int = 512, dim_text: int = 512, dim_inner_embed: int = 512):
        super(SaSR_Net, self).__init__()

        self.dim_audio: int = dim_audio
        self.dim_visual: int = dim_visual
        self.dim_test: int = dim_text
        self.dim_inner_embed: int = dim_inner_embed

        # for features
        self.fc_a1 = nn.Linear(dim_audio, dim_inner_embed)
        self.fc_a2 = nn.Linear(dim_inner_embed, dim_inner_embed)

        self.visual_net = resnet18(pretrained=True)

        self.fc_fusion = nn.Linear(dim_inner_embed * 2, dim_inner_embed)

        self.linear11 = nn.Linear(dim_inner_embed, dim_inner_embed)
        self.dropout1 = nn.Dropout(0.1)
        self.linear12 = nn.Linear(dim_inner_embed, dim_inner_embed)

        self.linear21 = nn.Linear(dim_inner_embed, dim_inner_embed)
        self.dropout2 = nn.Dropout(0.1)
        self.linear22 = nn.Linear(dim_inner_embed, dim_inner_embed)
        self.norm1 = nn.LayerNorm(dim_inner_embed)
        self.norm2 = nn.LayerNorm(dim_inner_embed)
        self.dropout3 = nn.Dropout(0.1)
        self.dropout4 = nn.Dropout(0.1)
        self.norm3 = nn.LayerNorm(dim_inner_embed)

        self.attn_a = nn.MultiheadAttention(dim_inner_embed, 4, dropout=0.1)
        self.attn_v = nn.MultiheadAttention(dim_inner_embed, 4, dropout=0.1)

        # question
        self.question_encoder = QstEncoder(
            93, dim_inner_embed, dim_inner_embed, 1, dim_inner_embed)

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.fc_ans = nn.Linear(dim_inner_embed, 42)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_gl = nn.Linear(dim_inner_embed * 2, dim_inner_embed)

        # combine
        self.fc1 = nn.Linear(dim_inner_embed * 2, dim_inner_embed)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(dim_inner_embed, dim_inner_embed // 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(dim_inner_embed // 2, dim_inner_embed // 4)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(dim_inner_embed // 4, 2)
        self.relu4 = nn.ReLU()

        # sasr
        self.sasr = SourceAwareSemanticRepresentation()

    def forward(self, audio, visual_posi, visual_nega, question):
        '''
            input question shape:    [B, T]
            input audio shape:       [B, T, C]
            input visual_posi shape: [B, T, C, H, W]
            input visual_nega shape: [B, T, C, H, W]
        '''

        # question features
        qst_feature = self.question_encoder(question)
        xq = qst_feature.unsqueeze(0)

        ###############################################################################################
        # visual posi

        audio_feat_posi, visual_feat_grd_posi, out_match_posi, av_cls_prob_posi, v_prob_posi, a_prob_posi, mask_posi = self.out_match_infer(
            audio, visual_posi)

        ###############################################################################################

        ###############################################################################################
        # visual nega

        audio_feat_nega, visual_feat_grd_nega, out_match_nega, av_cls_prob_nega, v_prob_nega, a_prob_nega, mask_posi = self.out_match_infer(
            audio, visual_nega)

        ###############################################################################################

        B = xq.shape[1]
        visual_feat_grd_be = visual_feat_grd_posi.view(
            B, -1, self.dim_inner_embed)   # [B, T, 512]
        visual_feat_grd = visual_feat_grd_be.permute(1, 0, 2)

        # attention, question as query on visual_feat_grd
        visual_feat_att = self.attn_v(
            xq, visual_feat_grd, visual_feat_grd, attn_mask=None, key_padding_mask=None)[0].squeeze(0)
        src = self.linear12(self.dropout1(
            F.relu(self.linear11(visual_feat_att))))
        visual_feat_att = visual_feat_att + self.dropout2(src)
        visual_feat_att = self.norm1(visual_feat_att)

        # attention, question as query on audio
        audio_feat_be = audio_feat_posi.view(B, -1, self.dim_inner_embed)
        audio_feat = audio_feat_be.permute(1, 0, 2)
        audio_feat_att = self.attn_a(
            xq, audio_feat, audio_feat, attn_mask=None, key_padding_mask=None)[0].squeeze(0)
        src = self.linear22(self.dropout3(
            F.relu(self.linear21(audio_feat_att))))
        audio_feat_att = audio_feat_att + self.dropout4(src)
        audio_feat_att = self.norm2(audio_feat_att)

        feat = torch.cat((audio_feat_att+audio_feat_be.mean(dim=-2).squeeze(),
                         visual_feat_att+visual_feat_grd_be.mean(dim=-2).squeeze()), dim=-1)
        feat = self.tanh(feat)
        feat = self.fc_fusion(feat)

        # fusion with question
        combined_feature = torch.mul(feat, qst_feature)
        combined_feature = self.tanh(combined_feature)
        
        # [batch_size, ans_vocab_size]
        out_qa = self.fc_ans(combined_feature)

        return out_qa, out_match_posi, out_match_nega, av_cls_prob_posi, v_prob_posi, a_prob_posi, mask_posi

    def out_match_infer(self, audio, visual):

        # audio features  [2*B*T, 128]
        audio_feat = F.relu(self.fc_a1(audio))
        audio_feat = self.fc_a2(audio_feat)
        audio_feat_pure = audio_feat
        B, T, C = audio_feat.size()             # [B, T, C]
        audio_feat = audio_feat.view(B, T, C)    # [B*T, C]

        # visual posi [2*B*T, C, H, W]
        B, T, C, H, W = visual.size()
        temp_visual = visual.view(B*T, C, H, W)            # [B*T, C, H, W]
        # [B*T, C, 1, 1]
        v_feat = self.avgpool(temp_visual)
        visual_feat_before_grounding = v_feat.squeeze()    # [B*T, C]
        visual_feat_before_grounding = visual_feat_before_grounding.view(
            B, -1, C)

        _, _, av_cls_prob, a_prob, v_prob, grouped_audio_embedding, grouped_visual_embedding = self.sasr(
            audio_feat, visual_feat_before_grounding, visual_feat_before_grounding)

        (B, C, H, W) = temp_visual.size()
        # [B*T, C, HxW]
        v_feat = temp_visual.view(B, C, H * W)
        # [B, HxW, C]
        v_feat = v_feat.permute(0, 2, 1)
        visual_feat = nn.functional.normalize(v_feat, dim=2)   # [B, HxW, C]

        # audio-visual grounding posi
        (B, T, C) = grouped_audio_embedding.size()
        audio_feat_aa = grouped_audio_embedding.view(
            B * T, -1).unsqueeze(dim=-1)                   # [B*T, C, 1]
        audio_feat_aa = nn.functional.normalize(
            audio_feat_aa, dim=1)   # [B*T, C, 1]

        x2_va = torch.matmul(
            visual_feat, audio_feat_aa).squeeze()  # [B*T, HxW]

        # [B*T, 1, HxW]
        x2_p = F.softmax(x2_va, dim=-1).unsqueeze(-2)
        visual_feat_grd = torch.matmul(x2_p, visual_feat)
        # [B*T, C]
        visual_feat_grd_after_grounding = visual_feat_grd.squeeze()

        grouped_visual_embedding = grouped_visual_embedding.flatten(0, 1)
        visual_gl = torch.cat(
            (grouped_visual_embedding, visual_feat_grd_after_grounding), dim=-1)
        visual_feat_grd = self.tanh(visual_gl)
        visual_feat_grd = self.fc_gl(visual_feat_grd)              # [B*T, C]

        grouped_audio_embedding = grouped_audio_embedding.flatten(0, 1)

        # [B*T, C*2], [B*T, 1024]
        feat = torch.cat((grouped_audio_embedding, visual_feat_grd), dim=-1)

        feat = F.relu(self.fc1(feat))       # (1024, 512)
        feat = F.relu(self.fc2(feat))       # (512, 256)
        feat = F.relu(self.fc3(feat))       # (256, 128)
        out_match = self.fc4(feat)     # (128, 2)

        return grouped_audio_embedding, visual_feat_grd, out_match, av_cls_prob, v_prob, a_prob, x2_p
