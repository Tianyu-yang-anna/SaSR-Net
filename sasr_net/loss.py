import torch
import torch.nn as nn
import torch.nn.functional as F

# For typing
from torch import Tensor
from typing import *


class LossSemantic(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        loss: Tensor = F.binary_cross_entropy(pred.float(), target.float())
        return loss


class LossCorrelation(nn.Module):
    def __init__(self, num_class: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_class: int = num_class

    def forward(self, pred: Tensor) -> Tensor:
        target: Tensor = torch.arange(0, self.num_class).long().to(pred.device)
        loss: Tensor = F.cross_entropy(pred, target)
        return loss

class LossAVQA(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        loss: Tensor = F.cross_entropy(pred, target)
        return loss 
    
class LossAVMatch(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, out_match_posi: Tensor, out_match_nega: Tensor) -> Tensor:
        out_match_pred, out_match_target = self.get_out_match_pred_target(out_match_posi, out_match_nega) 
        loss: Tensor = F.cross_entropy(out_match_pred, out_match_target)
        return loss 
    
    def get_out_match_pred_target(self, out_match_posi: Tensor, out_match_nega: Tensor):
        device: torch.device = torch.device(out_match_posi.device)
        pred: Tensor = torch.zeros((out_match_posi.shape[0] * 2, out_match_posi.shape[1]), dtype=torch.float32, device=device)
        target: Tensor = torch.zeros(out_match_posi.shape[0] * 2, dtype=torch.int64, device=device)
        for i in range(out_match_posi.shape[0]):
            pred[i * 2, :] = out_match_posi[i, :]
            pred[i * 2 + 1, :] = out_match_nega[i, :]
            target[i * 2] = 1
            target[i * 2 + 1] = 0
        return pred, target
        
