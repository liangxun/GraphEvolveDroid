import os
import torch
import torch.nn.functional as F


def compute_mmd(source, target):
    """计算MMD损失
    """
    source = source.mean(dim=0, keepdim=True)
    target = target.mean(dim=0, keepdim=True)
    
    loss = F.mse_loss(source, target)
    return loss