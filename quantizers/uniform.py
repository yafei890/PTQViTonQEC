import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from quantizers._ste import *


class UniformQuantizer(nn.Module):
    def __init__(self, n_bits: int = 4, symmetric: bool = False, channel_wise: bool = False):
        super().__init__()
        self.symmetric = symmetric # 对称量化
        self.n_bits = n_bits
        self.n_levels = 2 ** (self.n_bits - 1) # 量化的上下界？# 量化级别数（2^(n_bits-1)，对称量化时使用）
        self.channel_wise = channel_wise # 怎么这个均匀量化器中没有体现通道量化？——确实没有，AdaLog中也没有，这仅仅是一个标志位
        self.drop_prob = 1.0
        self.inited = False # 这个参数怎么没看见有赋值？——还真是没有赋值，在全部的代码文件中都没找到
        self.training_mode = False # 是否在训练模式下使用 STE（Straight-Through Estimator）
        # self.use_clip_forward = False # ？

        # self.register_buffer('scale', torch.tensor(1.0))  # 占位初始化
        # self.register_buffer('zero_point', torch.tensor(0))

    def init_training(self):
        self.training_mode = True

    def end_training(self):
        self.training_mode = False

    def forward(self, x):
        if self.n_bits == 32:
            return x
        assert self.inited
        x_int = round_ste(x / self.scale) if self.training_mode else torch.round(x / self.scale)
        if self.symmetric: # 使用对称量化
            x_quant = x_int.clamp(-self.n_levels, self.n_levels - 1) # 在n_bits=8的情况下，clamp(-128，127)，这没问题
            x_dequant = x_quant * self.scale
        else: # 使用非对称量化
            x_quant = (x_int + round_ste(self.zero_point)).clamp(0, 2 * self.n_levels - 1) # 有数据的偏移；在n_bits=8的情况下，clamp(0，255)，这也没问题
            x_dequant = (x_quant - round_ste(self.zero_point)) * self.scale
        return x_dequant
    """ 2025年8月12日：重写这个forward方法，这个版本不支持梯度的后向传播，导致AdaRound用了也是白用
    def forward(self, x):
        if self.n_bits == 32:
            return x
        assert self.inited
        # print(f"x.device = {x.device}")
        # print(f"self.scale.device = {self.scale.device}")
        # sys.exit()
        x_int = torch.round(x / self.scale)
        if self.symmetric: # 使用对称量化
            x_quant = x_int.clamp(-self.n_levels, self.n_levels - 1)
            x_dequant = x_quant * self.scale
        else: # 使用非对称量化
            '''这里使用的是STE，并不是直接舍入，这是因为非对称量化时zero_point的舍入误差无法通过训练调整，对称量化就没有这个问题，因为对称量化的zp固定为0'''
            x_quant = (x_int + round_ste(self.zero_point)).clamp(0, 2 * self.n_levels - 1) # 有数据的偏移
            x_dequant = (x_quant - round_ste(self.zero_point)) * self.scale
        return x_dequant # 直接返回反量化后的值
        """

    def __repr__(self):
        return f'{self.__class__.__name__}(n_bits={self.n_bits}, symmetric={self.symmetric}, channel_wise={self.channel_wise})'


# class ShiftUniformQuantizer(UniformQuantizer):
#     def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False):
#         super().__init__(n_bits, symmetric, channel_wise)
#         self.shift = nn.Parameter(torch.zeros((1)))
#         self.register_buffer('bias_reparamed', torch.tensor(False))
#
#     def forward(self, x):
#         result = UniformQuantizer.forward(self, x + self.shift)
#         return result if self.bias_reparamed else result - self.shift

import torch
import torch.nn as nn


class _UniformQuantizer(nn.Module):
    def __init__(self,
                 n_bits: int = 8,
                 symmetric: bool = True,
                 channel_wise: bool = False,
                 channel_axis: int = -1,
                 ):
        super().__init__()
        self.n_bits = n_bits
        self.symmetric = symmetric
        self.channel_wise = channel_wise
        self.channel_axis = channel_axis
        self.drop_prob = 1.0

        # Quantization range
        self.qmin = -2 ** (n_bits - 1) if symmetric else 0
        self.qmax = 2 ** (n_bits - 1) - 1 if symmetric else 2 ** n_bits - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel_wise:
            # print(self.channel_axis)
            return self._quantize_per_channel(x) # 还是返回反量化后的值
        else:
            # print(self.channel_axis)
            return self._quantize_per_tensor(x) # 还是返回反量化后的值

    def _quantize_per_tensor(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate min/max
        if self.symmetric:
            abs_max = torch.max(torch.abs(x))
            scale = abs_max / self.qmax if abs_max > 0 else torch.tensor(1.0)
            zero_point = torch.zeros_like(scale)
        else:
            min_val = torch.min(x)
            max_val = torch.max(x)
            scale = (max_val - min_val) / (self.qmax - self.qmin)
            zero_point = self.qmin - min_val / scale

        # Quantize
        x_int = torch.round(x / scale + zero_point)
        x_int = torch.clamp(x_int, self.qmin, self.qmax)

        # return x_int # 暂时先不返回反量化后的值，先返回整数值

        # Dequantize
        x_dequant = (x_int - zero_point) * scale
        return x_dequant

    def _quantize_per_channel(self, x: torch.Tensor) -> torch.Tensor:
        # Move channel axis to last dim for easier processing
        x = x.transpose(self.channel_axis, -1)
        orig_shape = x.shape
        x = x.reshape(-1, x.shape[-1])  # [..., channel_dim]

        # Calculate scales/zero_points per channel
        if self.symmetric:
            abs_max = torch.max(torch.abs(x), dim=0)[0]
            scales = abs_max / self.qmax
            scales[abs_max == 0] = 1.0  # Avoid division by zero
            zero_points = torch.zeros_like(scales)
        else:
            min_vals = torch.min(x, dim=0)[0]
            max_vals = torch.max(x, dim=0)[0]
            scales = (max_vals - min_vals) / (self.qmax - self.qmin)
            zero_points = self.qmin - min_vals / scales

        # Quantize
        x_int = torch.round(x / scales + zero_points)
        x_int = torch.clamp(x_int, self.qmin, self.qmax)

        # x_int = x_int.reshape(orig_shape)
        # x_int = x_int.transpose(self.channel_axis, -1)
        # return x_int # 暂时先不返回反量化后的值，先返回整数值

        # Dequantize
        x_dequant = (x_int - zero_points) * scales

        # Restore original shape
        x_dequant = x_dequant.reshape(orig_shape)
        x_dequant = x_dequant.transpose(self.channel_axis, -1)
        return x_dequant

    def extra_repr(self) -> str:
        return f"n_bits={self.n_bits}, symmetric={self.symmetric}, per_channel={self.channel_wise}, channel_axis={self.channel_axis}"


class TwinUniformQuantizer(UniformQuantizer):
    """这个先留着，后面对GELU激活函数可以试一试"""
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False):
        super().__init__(n_bits, symmetric, channel_wise)
    
    def forward(self, x):
        if self.n_bits == 32:
            return x # 无需量化
        assert self.inited and self.scale.shape[0] == 2 # 是否已经初始化且有两个scale参数
        if self.training_mode: # 训练模式
            '''pos是正数的处理，neg是负数的处理。两个scale，分开量化，也没什么复杂的。
            训练模式使用STE方便传递梯度，推理模式直接进行舍入。
            '''
            x_pos = round_ste(x / (self.scale[0])).clamp(0, self.n_levels - 1).mul(self.scale[0]) # 正数限制在[0, n_levels-1]
            x_neg = round_ste(x / (self.scale[1])).clamp(-self.n_levels, 0).mul(self.scale[1]) # 负数限制在[-n_levels, 0]
        else: # 推理模式
            x_pos = torch.round(x / (self.scale[0])).clamp(0, self.n_levels - 1).mul(self.scale[0])
            x_neg = torch.round(x / (self.scale[1])).clamp(-self.n_levels, 0).mul(self.scale[1])
        x_dequant = (x_pos + x_neg).reshape_as(x) # 合并结果，并恢复到原来的形状
        return x_dequant
