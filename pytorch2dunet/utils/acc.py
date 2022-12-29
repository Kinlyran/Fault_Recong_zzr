import torch
from torch import Tensor

def acc_score(pred, true_mask):
    return torch.mean((pred==true_mask).float())
    