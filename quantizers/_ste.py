import torch
from torch import nn


def round_ste(x: torch.Tensor):
    return (x.round() - x).detach() + x
    """
    .detach() 剥离梯度，使得反向传播时被视为常数；
    +x：重新加上原始值 x；
    量化中的 round 操作是不可微分的，直接使用会导致梯度为 0，无法训练；因此STE解决的就是在前向传播时正常量化，在反向传播时假设量化操作是恒等映射（梯度为1），让梯度穿透量化层
    """



def floor_ste(x: torch.Tensor):
    return (x.floor() - x).detach() + x


def ceil_ste(x: torch.Tensor):
    return (x.ceil() - x).detach() + x
