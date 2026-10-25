from types import MethodType
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import logging
from timm.models import vision_transformer
from timm.models.vision_transformer import Attention
from timm.models.swin_transformer import WindowAttention

class MatMul(nn.Module):
    def forward(self, A, B):
        return A @ B

"""以下两个注意力的计算流程，主要是为了把两个矩阵乘法拆出来，然后对矩阵乘量化"""
def vit_attn_forward(self, x):
    # print(f"type of self.matmul1: {type(self.matmul1)}")
    # print(f"is self.matmul1 callable? {callable(self.matmul1)}")
    # print(f"type of self.scale: {type(self.scale)}")
    # print(f"value of self.scale: {self.scale}")

    # exit()
    B, N, C = x.shape
    x = self.qkv(x)
    qkv = x.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    # 交换之后：[3, B, num_heads, N, C//num_heads]
    # 再拆分QKV之后，每个矩阵维度都是：[B, num_heads, N, C//num_heads]
    # DeiT-Tiny：[32, 3, 197, 64]，也就是说，得把[32,197,192]的192拆分成64*3并且还要重排。
    q, k, v = (qkv[0], qkv[1], qkv[2])
    q, k = self.q_norm(q), self.k_norm(k) # 这里加了一个norm，不知道有什么用。这好像是原版模型自带的，而Yuan的代码才是自定义的
    attn = self.matmul1(q, k.transpose(-2, -1)) * self.scale # 这个缩放因子并不是量化参数，而是注意力中本身固有的参数
    # attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # 替换，测试是不是MinMaxQuantMatMul的问题

    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = self.matmul2(attn, v)
    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x
'''swin transformer'''
def swin_attn_forward(self, x, mask=None):
    B_, N, C = x.shape
    x = self.qkv(x)
    qkv = x.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q = q * self.scale
    attn = self.matmul1(q, k.transpose(-2, -1))
    attn = attn + self._get_rel_pos_bias()
    if mask is not None:
        nW = mask.shape[0]
        attn = attn.view(-1, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = self.matmul2(attn, v).transpose(1, 2).reshape(B_, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x

def get_net(name): # name是传进来的模型名称
    model = timm.create_model(name, pretrained=True) # 使用timm库根据传进来的name创建模型
    logging.info(f"HF Hub ID: {model.default_cfg["hf_hub_id"]}")  # 检查是否通过HF加载
    '''替换原来的Linear层'''
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            setattr(module, "matmul1", MatMul())
            setattr(module, "matmul2", MatMul())
            module.forward = MethodType(vit_attn_forward, module)
        if isinstance(module, WindowAttention):
            setattr(module, "matmul1", MatMul())
            setattr(module, "matmul2", MatMul())
            module.forward = MethodType(swin_attn_forward, module)
    return model
    '''AdaLog的这个方法中后面还有一部分，作用是逐层包装替换，但是我觉得暂时用不到，只保留上面替换矩阵乘法的代码'''



