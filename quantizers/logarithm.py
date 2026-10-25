import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from quantizers._ste import *


class Log2Quantizer(nn.Module):
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False):
        super().__init__()
        self.sym = symmetric
        self.n_bits = n_bits
        self.n_levels = 2 ** (self.n_bits - 1) # 8bit下是128；4bit是8
        self.inited = False
        self.drop_prob = 1.0
        self.channel_wise = channel_wise
        self.training_mode = False

    def init_training(self):
        self.training_mode = True

    def end_training(self):
        self.training_mode = False
        
    def forward(self, x):
        if self.n_bits == 32: # 不需要量化
            return x
        assert self.inited # 确保已经被初始化
        scaled_x = (x / self.scale).clamp(min=1e-15, max=1.0) # 下限是为了防止对0取对数，上限是保证对数结果为非正数（log2(1.0) = 0），因此下面要取符号？
        # 根据是否训练模式决定是否使用STE；log量化器应该都是要去一个负号。
        if self.training_mode:
            x_quant = round_ste(-scaled_x.log2())
        else:
            torch.round(-scaled_x.log2())
        mask = x_quant < 2 * self.n_levels # 乘法优先计算，生成一个bool类型的掩码张量，用于检查哪些值在量化范围内
        x_quant = torch.clamp(x_quant, 0, 2 * self.n_levels - 1) # 截断
        x_dequant = 2 ** (-1 * x_quant) * self.scale # 反量化
        x_dequant = x_dequant * mask # 乘上掩码，将超出量化范围的值置零从而去除不在量化级别范围的值；
        return x_dequant

    def __repr__(self):
        return f'{self.__class__.__name__}(n_bits={self.n_bits}, sym={self.sym}, channel_wise={self.channel_wise}, log_base={2})'


class LogSqrt2Quantizer(Log2Quantizer):
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False):
        super().__init__(n_bits, symmetric, channel_wise)
        
    def forward(self, x):
        if self.n_bits == 32:
            return x
        assert self.inited
        scaled_x = (x / self.scale).clamp(min=1e-15, max=1.0)
        if self.training_mode:
            x_quant = round_ste(-scaled_x.log2() * 2)
            mask = x_quant < 2 * self.n_levels
            x_quant = torch.clamp(x_quant, 0, 2 * self.n_levels - 1)
            x_dequant = 2 ** (-1 * x_quant / 2) * self.scale
        else:
            x_quant = torch.round(-scaled_x.log2() * 2)
            mask = x_quant < 2 * self.n_levels
            x_quant = torch.clamp(x_quant, 0, 2 * self.n_levels - 1)
            odd_mask = (x_quant % 2) * (math.sqrt(2) - 1) + 1 # 奇数量化级别修正，对应论文中的公式(4)，加1是为了保证非零
            '''反量化这里不应该是floor吗？怎么变成ceil了？好像是为了保证反量化后的值不会大于原始值，是一种保守估计'''
            '''这里没看明白，为什么要用ceil，论文中用的是floor'''
            x_dequant = 2 ** (-1 * torch.ceil(x_quant / 2)) * odd_mask * self.scale # log sqrt2量化器的反量化公式与log2量化器不一样，因此要分开判断。
        x_dequant = x_dequant * mask
        return x_dequant

    def __repr__(self):
        return f'{self.__class__.__name__}(n_bits={self.n_bits}, sym={self.sym}, channel_wise={self.channel_wise}, log_base={math.sqrt(2)})'
    
    
class AdaLogQuantizer(Log2Quantizer):
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False):
        super().__init__(n_bits, symmetric, channel_wise)
        self.r = 37.0 # 质数，论文公式中的参数，用来将小数转换成整数，方便硬件的移位操作
        self.register_buffer('q', torch.tensor([int(self.r)]))
        '''两个查找表，加速推理'''
        self.register_buffer('table1', torch.zeros((self.n_levels * 2)))
        self.register_buffer('table2', torch.zeros((self.n_levels * 2)))
        self.update_table() # 初始化table1和table2？

    def update_table(self):
        for i in range(0, self.n_levels * 2): # 这个i就是量化后的值，因为需要记录下来，所以对每个量化后的值都需要处理。
            '''table1:存储 floor(i * q / r)，用于快速计算指数部分。'''
            self.table1[i].data.copy_(torch.tensor(math.floor(i * self.q.item() / self.r))) # 公式8的前半部分
            '''table2:存储归一化后的值 val
            (4 * self.n_levels - 2)：这是一个经验性参数，平衡精度与存储。其中2*self.n_levels是总共的量化级别，该公式原本的形式是：2*(2*n_levels)-2
            这里也不明白为什么选取这个值。但是离散化操作（round(x*K)/K）是量化领域的常见技巧。
            '''
            val = round((2 ** (-((self.q.item() * i) % self.r) / self.r)) * (4 * self.n_levels - 2)) / (4 * self.n_levels - 2)
            self.table2[i].data.copy_(torch.tensor(val))
    def forward(self, x):
        if self.n_bits == 32:
            return x
        assert self.inited
        scaled_x = (x / self.scale).clamp(min=1e-15, max=1.0)
        '''参考论文中的公式(8)'''
        if self.training_mode: # 训练模式
            x_quant = round_ste(-scaled_x.log2() * self.r / self.q)
            mask = x_quant < 2 * self.n_levels
            x_quant = torch.clamp(x_quant, 0, 2 * self.n_levels - 1)
            x_dequant = 2 ** (-1 * x_quant * self.q / self.r) * self.scale # 训练的时候动态计算
        else: # 推理模式，这时候就不需要计算了，直接查表
            x_quant = torch.round(-scaled_x.log2() * self.r / self.q)
            mask = x_quant < 2 * self.n_levels
            x_quant = torch.clamp(x_quant, 0, 2 * self.n_levels - 1)
            x_dequant = (2 ** (-self.table1[x_quant.long()])) * self.table2[x_quant.long()] * self.scale # 两次查表，一次移位，硬件上都是比较方便实现的
        x_dequant = x_dequant * mask
        return x_dequant

    def __repr__(self):
        return f'{self.__class__.__name__}(n_bits={self.n_bits}, sym={self.sym}, channel_wise={self.channel_wise}, q={self.q.item()})'
        
'''下面这三个类主要是针对Swin Transformer的，引入了可学习的偏移量。
2025年6月4日：shift并不是针对swin的量化器，而是针对GELU函数，将其负值区间的值偏移到正值区间，偏移量大约是0.17
2025年6月25日：这个是我后续的工作，负值区间不需要偏移，而是应用双均匀量化'''
class ShiftLog2Quantizer(Log2Quantizer):
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False):
        super().__init__(n_bits, symmetric, channel_wise)
        self.shift = nn.Parameter(torch.zeros((1))) # 可学习参数
        self.register_buffer('bias_reparamed', torch.tensor(False)) # 缓冲区标志，控制是否在输出中补偿平移量，默认False

    def forward(self, x):
        result = Log2Quantizer.forward(self, x + self.shift) # 加上一个平移，然后再前向传播
        return result if self.bias_reparamed else result - self.shift


class ShiftLogSqrt2Quantizer(LogSqrt2Quantizer):
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False):
        super().__init__(n_bits, symmetric, channel_wise)
        self.shift = nn.Parameter(torch.zeros((1)))
        self.register_buffer('bias_reparamed', torch.tensor(False))

    def forward(self, x):
        result = LogSqrt2Quantizer.forward(self, x + self.shift)
        return result if self.bias_reparamed else result - self.shift


class ShiftAdaLogQuantizer(AdaLogQuantizer):
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False):
        super().__init__(n_bits, symmetric, channel_wise)
        self.shift = nn.Parameter(torch.zeros((1)))
        self.register_buffer('bias_reparamed', torch.tensor(False))

    def forward(self, x):
        # print(f"self.bias_reparamed = {self.bias_reparamed}")
        # print(f"运行到了ShiftAdaLogQuantizer类中")
        result = AdaLogQuantizer.forward(self, x + self.shift)
        return result if self.bias_reparamed else result - self.shift
        