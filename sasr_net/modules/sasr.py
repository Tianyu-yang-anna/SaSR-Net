from grouping import ModalityTrans
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from typing import *
from torch import Tensor
import sys

sys.path.append("./sasr_net")
sys.path.append("./sasr_net/models")


class SourceAwareSemanticRepresentation(nn.Module):
    """
    SourceAwareSemanticRepresentation is a neural network model for source-aware semantic representation.

    Args:
        dim_in (int, optional): Input dimension. Defaults to 512.
        dim_attn (int, optional): Dimension of attention. Defaults to 512.
        depth_vis (int, optional): Depth of visual layers. Defaults to 3.
        depth_aud (int, optional): Depth of audio layers. Defaults to 3.
        num_class (int, optional): Number of classes. Defaults to 21.
        unimodal_assgin (str, optional): Unimodal assignment type, "hard" or "soft". Defaults to "hard".

    Attributes:
        fc_a (nn.Linear): Linear layer for audio input.
        fc_v (nn.Linear): Linear layer for visual input.
        fc_st (nn.Linear): Linear layer for visual_st input.
        fc_fusion (nn.Linear): Linear layer for feature fusion.
        unimodal_assgin (str): Unimodal assignment type.
        audio_cug (ModalityTrans): Audio modality transformer.
        visual_cug (ModalityTrans): Visual modality transformer.
        fc_prob (nn.Linear): Linear layer for prediction.
        fc_prob_a (nn.Linear): Linear layer for audio prediction.
        fc_prob_v (nn.Linear): Linear layer for visual prediction.
        fc_cls (nn.Linear): Linear layer for class prediction.

    """

    def __init__(self, dim_in: int = 512, dim_attn: int = 512, depth_vis: int = 3, depth_aud: int = 3, num_class: int = 22, unimodal_assgin: str = "hard", *args: Any, **kwargs: Any):
        super(SourceAwareSemanticRepresentation, self).__init__()

        self.fc_a = nn.Linear(dim_in, dim_attn)
        self.fc_v = nn.Linear(dim_in, dim_attn)
        self.fc_st = nn.Linear(dim_in, dim_attn)
        self.fc_fusion = nn.Linear(dim_attn * 2, dim_attn)

        # hard or soft assignment
        self.unimodal_assgin = unimodal_assgin

        unimodal_hard_assignment = True

        # learnable tokens
        self.av_token = nn.Parameter(torch.zeros(num_class, dim_attn))

        # class-aware uni-modal grouping
        self.audio_cug = ModalityTrans(
            dim_attn,
            depth=depth_aud,
            num_heads=8,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop=0.,
            attn_drop=0.,
            drop_path=0.1,
            norm_layer=nn.LayerNorm,
            out_dim_grouping=dim_attn,
            num_heads_grouping=8,
            num_group_tokens=num_class,
            num_output_groups=num_class,
            hard_assignment=unimodal_hard_assignment,
            use_han=True
        )

        self.visual_cug = ModalityTrans(
            dim_attn,
            depth=depth_vis,
            num_heads=8,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop=0.,
            attn_drop=0.,
            drop_path=0.1,
            norm_layer=nn.LayerNorm,
            out_dim_grouping=dim_attn,
            num_heads_grouping=8,
            num_group_tokens=num_class,
            num_output_groups=num_class,
            hard_assignment=unimodal_hard_assignment,
            use_han=False
        )

        # prediction
        self.fc_prob = nn.Linear(dim_attn, 1)
        self.fc_prob_a = nn.Linear(dim_attn, 1)
        self.fc_prob_v = nn.Linear(dim_attn, 1)

        self.fc_cls = nn.Linear(dim_attn, num_class)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize model weights.

        Args:
            m (nn.Module): Model module.
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, audio: Tensor, visual: Tensor, visual_st: Tensor):
        """
        Forward pass of the model.

        Args:
            audio (torch.Tensor): Input audio data.
            visual (torch.Tensor): Input visual data.
            visual_st (torch.Tensor): Input visual_st data.

        Returns:
            torch.Tensor: Output for audio modality.
            torch.Tensor: Output for visual modality.
            torch.Tensor: Class probabilities.
            torch.Tensor: Global probabilities.
            torch.Tensor: Audio probabilities.
            torch.Tensor: Visual probabilities.
            torch.Tensor: Audio frame probabilities.
            torch.Tensor: Visual frame probabilities.
            torch.Tensor: Audio attention.
            torch.Tensor: Visual attention.
        """

        x1_0 = self.fc_a(audio)

        # 2d and 3d visual feature fusion
        vid_s = self.fc_v(visual)
        vid_st = self.fc_st(visual_st)
        x2_0 = torch.cat((vid_s, vid_st), dim=-1)
        x2_0 = self.fc_fusion(x2_0)

        # visual uni-modal groupingf
        x2, _, v_attn = self.visual_cug(
            x2_0, self.av_token, return_attn=True)

        # audio uni-modal grouping
        x1, _, a_attn = self.audio_cug(
            x1_0, self.av_token, x2_0, return_attn=True)

        # cls token prediction
        # [25, 25]
        av_cls_prob = self.fc_cls(self.av_token)

        # audio prediction
        # [B, 25, 1]
        a_prob = torch.sigmoid(self.fc_prob_a(x1))

        # [B, 25]
        a_prob = a_prob.sum(dim=-1)

        # visual prediction
        # [B, 25, 1]
        v_prob = torch.sigmoid(self.fc_prob_v(x2))

        # [B, 25]
        v_prob = v_prob.sum(dim=-1)
        return x1, x2, av_cls_prob, a_prob, v_prob, a_attn, v_attn
