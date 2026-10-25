import sys

import numpy as np
import torch
import torch.nn as nn
import copy
from quantizers._ste import *

class GPTQQuantizer(nn.Module):
    '''基础量化器，基础量化器不一定是简单量化器
    这个得仿照UniformQuantizer去写，将量化参数及量化参数的搜索过程要分离开。
    '''
    def __init__(
            self,
            shape: int = 1,
            n_bits: int = 8,
            channel_wise: bool = False,
            symmetric:bool=True,
            mse:bool=True,
            norm=2.4,
            grid=100,
            maxshrink=.8,
            blocksize=128,
            percdamp=.01,
            groupsize=-1,
            # actorder=False,
            actorder=False,
            static_groups=False

    ):
        # 配置参数
        self.n_bits = n_bits
        self.n_levels = 2 ** (self.n_bits - 1)
        self.channel_wise = channel_wise
        self.symmetric = symmetric
        self.mse = mse # 是否启用mse优化
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        self.Hessian_inited = False # 这个用来标志Hessian矩阵是否计算完成
        self.param_inited = False # 这个用来表示量化参数是否找到
        self.blocksize = blocksize
        self.percdamp = percdamp
        self.groupsize = groupsize
        self.actorder = actorder
        self.static_groups = static_groups
        self.nsamples = 0
        self.maxq = torch.tensor(2 ** self.n_bits - 1)
        super().__init__()  # 这几行代码是用来给父类量化器的量化参数赋值，删除的话就无法量化了。
        '''下面的属性和两个方法都是给重建准备的'''
        self.training_mode = False
        # self.register_buffer("rest", None)  # 存储量化时的舍入值，仅占位，维度在forward方法中确定。
        self.rest = None
        '''2025年9月1日14点33分：将Hessian矩阵放在量化器内部'''
        self.rows = None  # 行数,192。好像没用到
        self.cols = None  # 列数,768
        self.hessian = None
        self.current_index = None

    def set_index(self, index):
        self.current_index = index

    def init_training(self):
        self.training_mode = True

    def end_training(self):
        self.training_mode = False

    def forward(self, x):
        '''这里的inp就是权重，因为我只用GPTQ来量化权重。这个方法还没写完，因此不能用
        2025年8月13日：这个量化器的前向传播方法中不要再有搜索量化参数的代码了
        2025年8月21日：这个前向传播方法也只能处理一个子矩阵，无法对整个的权重矩阵进行量化'''
        '''这里还是有问题，attn.qkv传入的是子矩阵，因此rest统计的也是子矩阵的余数部分。那就在Linear层中再进行拼接，这里的处理就相当于inp_int和inp_sim'''
        # print("======================================================")
        H = self.hessian.data.clone()
        x_origin = x.clone()
        col = self.cols
        print(f"col={col}")
        dead = torch.diag(H) == 0  # 找出对角线为0的位置，也就是不可逆或者无信息的特征方向
        H[dead, dead] = 1
        x[:,dead] = 0
        Losses = torch.zeros_like(x)  # 统计损失
        inp_sim, inp_int = torch.zeros_like(x), torch.zeros_like(x)
        damp = self.percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(col)
        H[diag, diag] += damp  # 对角线元素加上阻尼，正定矩阵一定是方阵。
        H = torch.linalg.cholesky(H)  # Cholesky分解，得到下三角矩阵
        H = torch.cholesky_inverse(H)  # 通过分解后的形式求逆矩阵，相比直接求逆矩阵，分解后的数值更加稳定
        H = torch.linalg.cholesky(H,upper=True)  # 再对逆矩阵分解，得到上三角矩阵
        Hinv = H  # 改名，最后得到的是Hessian逆矩阵的Cholesky分解形式。
        # print(f"self.blocksize = {self.blocksize}")
        for i1 in range(0, col, self.blocksize):  # i1：0~columns，步长=blocksize
            i2 = min(i1 + self.blocksize, col)
            count = i2 - i1  # 这是计算当前block有多少列，防止列数不足blocksize的时候超过范围
            W1 = x[:, i1:i2].clone()  # 分块之后的权重子矩阵
            Q1 = torch.zeros_like(W1)  # 子矩阵的量化结果
            Err1 = torch.zeros_like(W1)  # 用于存储当前block内每一列量化时产生的误差向量，当前block量化结束后要把误差传播到其他block
            Losses1 = torch.zeros_like(W1)  # 存储当前block内每一列量化带来的近似损失，估算当前block的总误差
            Hinv1 = Hinv[i1:i2, i1:i2]  # 权重子矩阵的Hessian矩阵对应的行和列
            # print(f"Hinv1是否全零：{Hinv1.count_nonzero()==0}")
            for i in range(count):
                w = W1[:, i]  # 一列权重
                d = Hinv1[i, i]  # d是hessian逆中的元素，用于评估一列权重的重要性。
                if self.training_mode:
                    q = torch.round_ste(w.unsqueeze(1) / self.scale) + self.zero_point
                else:
                    q = torch.round(w.unsqueeze(1) / self.scale) + self.zero_point
                q = torch.clamp(q, min=0, max=self.maxq) # 非对称的截断
                q_sim = self.scale * (q - self.zero_point)
                q_sim = q_sim.flatten()
                '''计算舍入的余数'''
                Q1[:, i] = q_sim  # 量化输出
                Losses1[:, i] = (w - q_sim) ** 2 / d ** 2
                err1 = (w - q_sim) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))  # 更新当前block中剩余列的权重，对应论文中的公式4
                Err1[:, i] = err1  # 累积block内的的误差
            inp_sim[:, i1:i2] = Q1  # 合并不同block的量化结果
            Losses[:, i1:i2] = Losses1 / 2  # 这里可能是为了匹配均方误差损失函数的计算，虽然论文中没有除2，但是常用的MSE公式中是需要除2的。
            x[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])  # 在剩余block中传播误差并更新未来列的权重
        # print(f"量化前后的权重是否一样：{torch.equal(x_origin,x)}")
        return inp_sim


    def __repr__(self):
        return f'{self.__class__.__name__}(n_bits={self.n_bits}, symmetric={self.symmetric}, channel_wise={self.channel_wise})'
