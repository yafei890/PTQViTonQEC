import torch
from torch import nn
from quantizers.uniform import UniformQuantizer
from quantizers._ste import round_ste


class AdaRoundQuantizer(nn.Module):
    def __init__(self, uq: UniformQuantizer, weight_tensor: torch.Tensor, round_mode='learned_hard_sigmoid'):
        super().__init__()
        # copying all attributes from UniformQuantizer # 在均匀量化器的基础上复用一些参数，增加自适应舍入功能
        self.n_bits = uq.n_bits
        # self.n_levels = uq.n_levels
        self.n_levels = 2 ** (uq.n_bits - 1)
        self.channel_wise = uq.channel_wise
        self.sym = uq.symmetric
        self.scale = nn.Parameter(uq.scale)
        self.zero_point = nn.Parameter(uq.zero_point)
        
        self.round_mode = round_mode
        self.alpha = None
        self.soft_targets = False

        # params for sigmoid function # Sigmoid函数的参数（用于软化舍入决策）
        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2/3
        self.init_alpha(x=weight_tensor.clone()) # 下面有这个方法

    def init_alpha(self, x: torch.Tensor):
        if self.round_mode == 'learned_hard_sigmoid':
            x_floor = torch.floor(x / self.scale)
            rest = (x / self.scale) - x_floor  # rest of rounding [0, 1)
            alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest
            self.alpha = nn.Parameter(alpha)
        else:
            raise NotImplementedError
    def forward(self, x):
        if self.round_mode == 'nearest': # 四舍五入
            x_int = torch.round(x / self.scale)
        elif self.round_mode == 'nearest_ste': # 使用直通估计器的四舍五入
            x_int = round_ste(x / self.scale) # 这是_ste.py中的方法
        elif self.round_mode == 'learned_hard_sigmoid': # 通过可学习的参数 alpha 决定舍入方向，自适应舍入
            x_floor = torch.floor(x / self.scale)
            if self.soft_targets: # 训练时可用软目标提升梯度稳定性，推理时用硬判决保证效率。
                x_int = x_floor + self.get_soft_targets()
            else:
                x_int = x_floor + (self.alpha >= 0).float()
        else:
            raise ValueError('Wrong rounding mode')
        if self.sym: # 对称量化？
            x_quant = torch.clamp(x_int, -self.n_levels, self.n_levels - 1)
            x_float_q = x_quant * self.scale
        else: # 非对称量化
            x_quant = torch.clamp(x_int + self.zero_point, 0, 2 * self.n_levels - 1)
            x_float_q = (x_quant - self.zero_point) * self.scale
        return x_float_q

    def get_soft_targets(self):
        return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)
    def get_hard_value(self, x):
        init_shape = x.shape
        return ((torch.floor(x.reshape_as(self.alpha) / self.scale) + (self.alpha >= 0).float()) * self.scale).reshape(*init_shape)



    def __repr__(self):
        return f'{self.__class__.__name__}(n_bits={self.n_bits}, sym={self.sym}, channel_wise={self.channel_wise}, round_mode={self.round_mode})'
