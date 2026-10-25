import sys

import torch
from torch import nn
from quantizers.gptq import GPTQQuantizer
from quantizers._ste import round_ste
import torch
from torch import nn
from quantizers.gptq import GPTQQuantizer
from torch.nn import functional as F

# class AdaRoundGPTQQuantizer(nn.Module):
#     def __init__(
#             self,
#             shape: int = 1,
#             n_bits: int = 8,
#             channel_wise: bool = False,
#             symmetric: bool = True,
#             mse: bool = True,
#             norm=2.4,
#             grid=100,
#             maxshrink=.8,
#             blocksize=128,
#             percdamp=.01,
#             groupsize=-1,
#             actorder=False,
#             static_groups=False,
#             round_mode='learned_hard_sigmoid'  # 新增舍入模式参数
#     ):
#         super().__init__()
#         self.n_bits = n_bits
#         self.n_levels = 2 ** (self.n_bits - 1)
#         self.channel_wise = channel_wise
#         self.symmetric = symmetric
#         self.mse = mse
#         self.norm = norm
#         self.grid = grid
#         self.maxshrink = maxshrink
#         self.Hessian_inited = False
#         self.param_inited = False
#         self.blocksize = blocksize
#         self.percdamp = percdamp
#         self.groupsize = groupsize
#         self.actorder = actorder
#         self.static_groups = static_groups
#         self.nsamples = 0
#         self.maxq = torch.tensor(2 ** self.n_bits - 1)
#
#         # 新增AdaRound相关参数
#         self.round_mode = round_mode
#         self.alpha = None
#         self.soft_targets = False
#         self.gamma, self.zeta = -0.1, 1.1
#         self.beta = 2 / 3
#
#         # 量化参数
#         self.scale = nn.Parameter(torch.ones(shape))
#         self.zero_point = nn.Parameter(torch.zeros(shape))
#
#     def init_alpha(self, x: torch.Tensor):
#         x_floor = torch.floor(x / self.scale)
#         if self.round_mode == 'learned_hard_sigmoid':
#             rest = (x / self.scale) - x_floor  # rest of rounding [0, 1)
#             alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest
#             self.alpha = nn.Parameter(alpha)
#         else:
#             raise NotImplementedError
#
#     def get_soft_targets(self):
#         return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)
#     def get_hard_value(self, x):
#         init_shape = x.shape
#         return ((torch.floor(x.reshape_as(self.alpha) / self.scale) + (self.alpha >= 0).float()) * self.scale).reshape(*init_shape)
#     def forward(self, inp, H, col):
#         dead = torch.diag(H) == 0
#         H[dead, dead] = 1
#         inp[:, dead] = 0
#         Losses = torch.zeros_like(inp)
#         inp_sim, inp_int = torch.zeros_like(inp), torch.zeros_like(inp)
#         damp = self.percdamp * torch.mean(torch.diag(H))
#         diag = torch.arange(col)
#         H[diag, diag] += damp
#         H = torch.linalg.cholesky(H)
#         H = torch.cholesky_inverse(H)
#         H = torch.linalg.cholesky(H)
#         Hinv = H
#
#         # 初始化alpha参数
#         if self.alpha is None:
#             self.init_alpha(inp)
#
#         for i1 in range(0, col, self.blocksize):
#             i2 = min(i1 + self.blocksize, col)
#             count = i2 - i1
#             W1 = inp[:, i1:i2].clone()
#             Q1 = torch.zeros_like(W1)
#             Err1 = torch.zeros_like(W1)
#             Losses1 = torch.zeros_like(W1)
#             Hinv1 = Hinv[i1:i2, i1:i2]
#
#             for i in range(count):
#                 w = W1[:, i]
#                 d = Hinv1[i, i]
#
#                 # 修改量化部分，支持不同舍入模式
#                 if self.round_mode == 'nearest':
#                     q = torch.round(w.unsqueeze(1) / self.scale) + self.zero_point
#                 elif self.round_mode == 'nearest_ste':
#                     q = round_ste(w.unsqueeze(1) / self.scale) + self.zero_point
#                 elif self.round_mode == 'learned_hard_sigmoid':
#                     x_floor = torch.floor(w.unsqueeze(1) / self.scale)
#                     if self.soft_targets:
#                         q = x_floor + self.get_soft_targets() + self.zero_point
#                     else:
#                         q = x_floor + (self.alpha >= 0).float() + self.zero_point
#                 else:
#                     raise ValueError('Wrong rounding mode')
#
#                 q = torch.clamp(q, min=0, max=self.maxq)
#                 q_sim = self.scale * (q - self.zero_point)
#                 q_sim = q_sim.flatten()
#                 Q1[:, i] = q_sim
#                 Losses1[:, i] = (w - q_sim) ** 2 / d ** 2
#                 err1 = (w - q_sim) / d
#                 W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
#                 Err1[:, i] = err1
#
#             inp_sim[:, i1:i2] = Q1
#             Losses[:, i1:i2] = Losses1 / 2
#             inp[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
#
#         return inp_sim
#
#     def __repr__(self):
#         return (f'{self.__class__.__name__}(n_bits={self.n_bits}, symmetric={self.symmetric}, '
#                 f'channel_wise={self.channel_wise}, round_mode={self.round_mode})')



class AdaRoundGPTQQuantizer(nn.Module):
    def __init__(self,quantizer: GPTQQuantizer, weight_tensor: torch.Tensor,round_mode="learned_hard_sigmoid"):
        super().__init__()
        self.hessian = quantizer.hessian
        self.cols = quantizer.cols
        self.n_bits = quantizer.n_bits
        # self.n_levels = uq.n_levels
        self.maxq = 2 ** (quantizer.n_bits - 1)
        self.channel_wise = quantizer.channel_wise # 默认通道量化，这个应该是用不到的
        self.symmetric = quantizer.symmetric # 默认非对称量化
        self.scale = nn.Parameter(quantizer.scale)
        self.zero_point = nn.Parameter(quantizer.zero_point)
        self.percdamp = quantizer.percdamp
        self.blocksize = quantizer.blocksize

        self.round_mode = round_mode
        self.alpha = None
        self.soft_targets = False

        # params for sigmoid function # Sigmoid函数的参数（用于软化舍入决策）
        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2 / 3
        self.init_alpha(W = weight_tensor.clone())  # 下面有这个方法的代码，
        self.current_index = None
    def set_index(self, index):
        self.current_index = index

    def init_alpha(self, W: torch.Tensor):
        '''通过self.scale.shape来判断卷积层还是Linear层'''
        if len(self.scale.shape) == 2: # 卷积层
            W = W.flatten(1)  # 先展平,无论Conv还是Linear都得按照第1维展平
            cols = W.shape[1]
            Rest = torch.zeros_like(W)  # 与w同形状的矩阵，保存每个权重元素的舍入值
            H = self.hessian.data.clone()
            dead = torch.diag(H) == 0  # 找出对角线为0的位置，也就是不可逆或者无信息的特征方向
            H[dead, dead] = 1
            W[:, dead] = 0
            Losses = torch.zeros_like(W)  # 统计损失
            damp = self.percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(cols)
            # H = H.clone()
            H[diag, diag] = H[diag, diag] + damp  # 对角线元素加上阻尼，正定矩阵一定是方阵。
            H = torch.linalg.cholesky(H)  # Cholesky分解，得到下三角矩阵
            H = torch.cholesky_inverse(H)  # 通过分解后的形式求逆矩阵，相比直接求逆矩阵，分解后的数值更加稳定
            H = torch.linalg.cholesky(H)  # 再对逆矩阵分解，得到上三角矩阵
            Hinv = H  # 改名，最后得到的是Hessian逆矩阵的Cholesky分解形式。
            for i1 in range(0, cols, self.blocksize):  # i1：0~columns，步长=blocksize
                i2 = min(i1 + self.blocksize, cols)
                count = i2 - i1  # 这是计算当前block有多少列，防止列数不足blocksize的时候超过范围
                W1 = W[:, i1:i2].clone()  # 分块之后的权重子矩阵
                Err1 = torch.zeros_like(W1)  # 用于存储当前block内每一列量化时产生的误差向量，当前block量化结束后要把误差传播到其他block
                Losses1 = torch.zeros_like(W1)  # 存储当前block内每一列量化带来的近似损失，估算当前block的总误差
                Hinv1 = Hinv[i1:i2, i1:i2]  # 权重子矩阵的Hessian矩阵对应的行和列
                R = torch.zeros_like(W1)  # 分块的舍入余数矩阵
                for i in range(count):
                    w = W1[:, i]  # 一列权重
                    d = Hinv1[i, i]  # d是hessian逆中的元素，用于评估一列权重的重要性。
                    w_floor = torch.floor(w.unsqueeze(1) / self.scale)
                    r = w.unsqueeze(1) / self.scale - w_floor  # 一列权重的余数
                    # 下面的代码之所以保留是为了计算损失，更新剩余未量化的权重
                    q = round_ste(w.unsqueeze(1) / self.scale) + self.zero_point
                    q = torch.clamp(q, min=0, max=self.maxq)  # 非对称的截断
                    q_sim = self.scale * (q - self.zero_point)
                    q_sim = q_sim.flatten()
                    Losses1[:, i] = (w - q_sim) ** 2 / d ** 2
                    err1 = (w - q_sim) / d
                    W1[:, i:] = W1[:, i:] - err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))  # 更新当前block中剩余列的权重，对应论文中的公式4
                    Err1[:, i] = err1  # 累积block内的的误差
                    # print(f"R[:,i].shape = {R[:,i].shape}")
                    # print(f"r.shape = {r.shape}")
                    R[:, i] = r.squeeze(1)
                Losses[:, i1:i2] = Losses1 / 2  # 这里可能是为了匹配均方误差损失函数的计算，虽然论文中没有除2，但是常用的MSE公式中是需要除2的。
                W[:, i2:] = W[:, i2:] - Err1.matmul(Hinv[i1:i2, i2:])  # 在剩余block中传播误差并更新未来列的权重
                Rest[:, i1:i2] = R
            '''计算完权重的舍入值rest后，初始化alpha'''
            if self.round_mode == 'learned_hard_sigmoid':
                alpha = -torch.log((self.zeta - self.gamma) / (Rest - self.gamma) - 1)
                self.alpha = nn.Parameter(alpha)
                # print(f"卷积层的alpha.shape = {self.alpha.shape}")
                # print(f"卷积层的量化参数：scale={self.scale.shape}，zp={self.zero_point.shape}")
                # sys.exit(789)
            else:
                raise NotImplementedError
        elif len(self.scale.shape) == 3:  # Linear层的处理，传入的时候已经对权重按照n_V进行了切分
            # print(f"W.shape: {W.shape}")  # [n_V,out_f//n_V,in_f]=[3,192,192] or [1,768,192]
            # print(f"self.scale.shape: {self.scale.shape}")  # [3,192,1]
            # print(f"self.zero_point.shape: {self.zero_point.shape}")  # [3,192,1]
            n_V = W.shape[0]
            cols = W.shape[2]  # W的列
            '''先把Hessian矩阵处理完'''
            H = self.hessian.data.clone()

            dead = torch.diag(H) == 0
            H[dead, dead] = 1
            damp = self.percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(cols)
            H[diag, diag] = H[diag, diag] + damp  # 对角线元素加上阻尼，正定矩阵一定是方阵。
            H = torch.linalg.cholesky(H)  # Cholesky分解，得到下三角矩阵
            H = torch.cholesky_inverse(H)  # 通过分解后的形式求逆矩阵，相比直接求逆矩阵，分解后的数值更加稳定
            H = torch.linalg.cholesky(H)  # 再对逆矩阵分解，得到上三角矩阵
            Hinv = H  # 改名，最后得到的是Hessian逆矩阵的Cholesky分解形式。
            Hinv_backup = H.clone()  # 这是为了防止量化一个权重子矩阵时对Hessian矩阵进行更新，从而影响后面的子矩阵量化
            '''然后是初始化需要用到的张量'''
            cols = W.shape[2]  # W的列
            # Rest = torch.zeros_like(W)  # 与w同形状的矩阵，保存每个权重元素的舍入值
            REST = [] # REST不应该初始化成张量，而是像接受量化权重那样初始化成一个列表，拼接子矩阵的结果，最后再处理维度。
            # Losses = torch.zeros_like(W)  # 统计损失 # 现在Losses统计的是整个权重矩阵，这是不对的。
            W_backup = W.clone()
            scale_backup = self.scale.detach().clone()
            zero_backup = self.zero_point.detach().clone()

            if n_V == 3: # attn.qkv

                for i in range(3):
                    # print(f"===============i={i}=========================")
                    self.scale = nn.Parameter(scale_backup[i])
                    self.zero_point = nn.Parameter(zero_backup[i])
                    W = W_backup[i] # 拿到一个子矩阵，这里已经没有第0维的n_V了，二维矩阵。
                    Losses = torch.zeros_like(W)  # 这里的Losses就是统计的子矩阵里面的损失
                    W[:, dead] = 0 # 子矩阵对应的元素置零
                    Rest = torch.zeros_like(W) # 这是子矩阵对应的余数子矩阵
                    Hinv = Hinv_backup.clone() # 这样就能保证每个权重子矩阵使用的都是未被修改的Hessian矩阵。
                    for i1 in range(0, cols, self.blocksize):  # i1：0~columns，步长=blocksize
                        i2 = min(i1 + self.blocksize, cols)
                        count = i2 - i1  # 这是计算当前block有多少列，防止列数不足blocksize的时候超过范围
                        W1 = W[:, i1:i2].clone()  # 分块之后的权重子矩阵
                        Err1 = torch.zeros_like(W1)  # 用于存储当前block内每一列量化时产生的误差向量，当前block量化结束后要把误差传播到其他block
                        Losses1 = torch.zeros_like(W1)  # 存储当前block内每一列量化带来的近似损失，估算当前block的总误差
                        Hinv1 = Hinv[i1:i2, i1:i2]  # 权重子矩阵的Hessian矩阵对应的行和列
                        R = torch.zeros_like(W1)  # 分块的舍入余数矩阵
                        for i in range(count):
                            w = W1[:, i]  # 一列权重
                            d = Hinv1[i, i]  # d是hessian逆中的元素，用于评估一列权重的重要性。
                            w_floor = torch.floor(w.unsqueeze(1) / self.scale)
                            r = w.unsqueeze(1) / self.scale - w_floor  # 一列权重的余数
                            # 下面的代码之所以保留是为了计算损失，更新剩余未量化的权重
                            q = round_ste(w.unsqueeze(1) / self.scale) + self.zero_point
                            q = torch.clamp(q, min=0, max=self.maxq)  # 非对称的截断
                            q_sim = self.scale * (q - self.zero_point)
                            q_sim = q_sim.flatten()
                            Losses1[:, i] = (w - q_sim) ** 2 / d ** 2
                            err1 = (w - q_sim) / d
                            W1[:, i:] = W1[:, i:] - err1.unsqueeze(1).matmul(
                            Hinv1[i, i:].unsqueeze(0))  # 更新当前block中剩余列的权重，对应论文中的公式4
                            Err1[:, i] = err1  # 累积block内的的误差
                            R[:, i] = r.squeeze(1)
                        Losses[:, i1:i2] = Losses1 / 2  # 这里可能是为了匹配均方误差损失函数的计算，虽然论文中没有除2，但是常用的MSE公式中是需要除2的。
                        W[:, i2:] = W[:, i2:] - Err1.matmul(Hinv[i1:i2, i2:])  # 在剩余block中传播误差并更新未来列的权重
                        Rest[:, i1:i2] = R
                    REST.append(Rest) # 拼接一个子矩阵的结果
                # 到这里，三个子矩阵就拼接完了，然后处理维度就行了。
                REST = torch.cat(REST, dim=0)
            elif n_V==1:
                W = W.squeeze(0) # 直接把n_V的维度压缩掉，就相当于是一个子矩阵了
                self.scale = nn.Parameter(scale_backup.squeeze(0))
                self.zero_point = nn.Parameter(zero_backup.squeeze(0))
                REST = torch.zeros_like(W) # 与权重同维度的舍入值矩阵
                cols = W.shape[1]
                H = self.hessian.data.clone()
                dead = torch.diag(H) == 0  # 找出对角线为0的位置，也就是不可逆或者无信息的特征方向
                H[dead, dead] = 1
                W[:, dead] = 0
                Losses = torch.zeros_like(W)  # 统计损失
                damp = self.percdamp * torch.mean(torch.diag(H))
                diag = torch.arange(cols)
                # H = H.clone()
                H[diag, diag] = H[diag, diag] + damp  # 对角线元素加上阻尼，正定矩阵一定是方阵。
                H = torch.linalg.cholesky(H)  # Cholesky分解，得到下三角矩阵
                H = torch.cholesky_inverse(H)  # 通过分解后的形式求逆矩阵，相比直接求逆矩阵，分解后的数值更加稳定
                H = torch.linalg.cholesky(H)  # 再对逆矩阵分解，得到上三角矩阵
                Hinv = H  # 改名，最后得到的是Hessian逆矩阵的Cholesky分解形式。
                for i1 in range(0, cols, self.blocksize):  # i1：0~columns，步长=blocksize
                    i2 = min(i1 + self.blocksize, cols)
                    count = i2 - i1  # 这是计算当前block有多少列，防止列数不足blocksize的时候超过范围
                    W1 = W[:, i1:i2].clone()  # 分块之后的权重子矩阵
                    Err1 = torch.zeros_like(W1)  # 用于存储当前block内每一列量化时产生的误差向量，当前block量化结束后要把误差传播到其他block
                    Losses1 = torch.zeros_like(W1)  # 存储当前block内每一列量化带来的近似损失，估算当前block的总误差
                    Hinv1 = Hinv[i1:i2, i1:i2]  # 权重子矩阵的Hessian矩阵对应的行和列
                    R = torch.zeros_like(W1)  # 分块的舍入余数矩阵
                    for i in range(count):
                        w = W1[:, i]  # 一列权重
                        d = Hinv1[i, i]  # d是hessian逆中的元素，用于评估一列权重的重要性。
                        w_floor = torch.floor(w.unsqueeze(1) / self.scale)
                        r = w.unsqueeze(1) / self.scale - w_floor  # 一列权重的余数
                        # 下面的代码之所以保留是为了计算损失，更新剩余未量化的权重
                        q = round_ste(w.unsqueeze(1) / self.scale) + self.zero_point
                        q = torch.clamp(q, min=0, max=self.maxq)  # 非对称的截断
                        q_sim = self.scale * (q - self.zero_point)
                        q_sim = q_sim.flatten()
                        Losses1[:, i] = (w - q_sim) ** 2 / d ** 2
                        err1 = (w - q_sim) / d
                        W1[:, i:] = W1[:, i:] - err1.unsqueeze(1).matmul(
                        Hinv1[i, i:].unsqueeze(0))  # 更新当前block中剩余列的权重，对应论文中的公式4
                        Err1[:, i] = err1  # 累积block内的的误差
                        # print(f"R[:,i].shape = {R[:,i].shape}")
                        # print(f"r.shape = {r.shape}") # [192,1]
                        # sys.exit("报错")
                        R[:, i] = r.squeeze(1)
                    Losses[:, i1:i2] = Losses1 / 2  # 这里可能是为了匹配均方误差损失函数的计算，虽然论文中没有除2，但是常用的MSE公式中是需要除2的。
                    W[:, i2:] = W[:, i2:] - Err1.matmul(Hinv[i1:i2, i2:])  # 在剩余block中传播误差并更新未来列的权重
                    REST[:, i1:i2] = R
                # print(f"REST.shape {REST.shape}")
                # sys.exit(1111)
            else:
                raise NotImplementedError("Linear层的n_V匹配错误：n_V != 3 and n_V != 1")
            '''恢复缩放因子和零点的形状'''
            self.scale = nn.Parameter(scale_backup)
            self.zero_point = nn.Parameter(zero_backup)

            '''计算完Linear层权重的舍入值REST后，初始化alpha'''
            if self.round_mode == 'learned_hard_sigmoid':
                alpha = -torch.log((self.zeta - self.gamma) / (REST - self.gamma) - 1)
                self.alpha = nn.Parameter(alpha)
                self.alpha = nn.Parameter(self.alpha.reshape_as(W_backup))
            else:
                raise NotImplementedError




        else:
            raise NotImplementedError("scale维度匹配错误：len(scale) != 3 and len(scale) != 1")

    '''2025年9月1日19点42分备份的代码，这是adalog中计算的方式，对于gptq算法来说，这种计算方式不适用。需要按照GPTQ算法的计算特性重新写。'''
    # def init_alpha(self, w: torch.Tensor):
    #     """初始化 alpha，使得 sigmoid(alpha) 与 weight 的小数部分对齐
    #     现在的计算方式是对整个张量进行计算的，我需要做的是拆分。那就得判断是卷积还是全连接，如果是全连接，还需要判断是否attn.qkv
    #     这个方法好像已经暗含了通道级的处理？"""
    #     print(f"init_alpha的self是什么？{self}，{type(self).__name__}") # 拿到的是AdaRoundGPTQQuantizer，拿不到层的信息
    #     # 能不能根据scale的形状判断呢？比如说，如果是卷积层，那么scale就是192，1；如果是全连接层，就可能是3，192，1或者1，192，1
    #     print(f"init_alpha中的scale.shape={self.scale.shape}") # shape=[192,1]
    #     # print(f"init_alpha方法中能拿到Hessian矩阵吗？self.H.shape = {self.H.shape}") # 拿不到Hessian矩阵，因为Hessian矩阵是保存在量化层中的，并不是保存在量化器中。
    #     if len(self.scale.shape) == 2: # 说明是卷积层
    #         '''卷积层可以先不管，就按照这种方式进行计算，在量化的时候再进行特定方式的取值'''
    #         w_floor = torch.floor(w / self.scale) # 这是用整个权重矩阵计算的
    #         if self.round_mode == 'learned_hard_sigmoid':
    #             rest = (w / self.scale) - w_floor  # rest of rounding [0, 1)
    #             alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest
    #             self.alpha = nn.Parameter(alpha)
    #         else:
    #             raise NotImplementedError
    #     elif len(self.scale.shape) == 3:  # 全连接层，需要按照scale的第一维拆分成子矩阵
    #         '''全连接层需要看情况进行拆分'''
    #         # 统一处理两种情况，对权重进行划分
    #         if self.scale.shape[0] == 3:  # 对应attn.qkv层
    #             w = w.reshape(3, -1, w.shape[1])  # [n_V,out_f//n_V,in_f]
    #         elif self.scale.shape[0] == 1:  # 对应其他Linear层
    #             w = w.unsqueeze(0)  # [1,out_f,in_f]
    #         else:
    #             raise NotImplementedError("Linear层匹配错误")
    #         # 统一处理计算逻辑
    #         w_floor = torch.floor(w / self.scale)
    #         if self.round_mode == 'learned_hard_sigmoid':
    #             rest = (w / self.scale) - w_floor  # rest of rounding [0, 1)
    #             alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest
    #             self.alpha = nn.Parameter(alpha)
    #         else:
    #             raise NotImplementedError
    #     else:
    #         raise NotImplementedError("scale维度匹配错误")



    # def get_soft_targets(self,i):
    #     """ Sigmoid 松弛，得到 [0,1] 的软舍入值 """
    #     res = torch.clamp(torch.sigmoid(self.alpha[:,i]) * (self.zeta - self.gamma) + self.gamma, 0, 1)
    #     return res
    def get_soft_targets(self, i=None):
        if i is None:
            res = torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)
        else:
            if len(self.alpha.shape) == 3: # 说明是Linear层
                if self.current_index is not None: # 说明是attn.qkv
                    res = torch.clamp(torch.sigmoid(self.alpha[self.current_index, :, i]) * (self.zeta - self.gamma) + self.gamma, 0, 1)
                else: # 说明是其他的Linear层
                    res = torch.clamp(torch.sigmoid(self.alpha[0, :, i]) * (self.zeta - self.gamma) + self.gamma, 0, 1)
            elif len(self.alpha.shape) == 2: # 卷积层
                res = torch.clamp(torch.sigmoid(self.alpha[:, i]) * (self.zeta - self.gamma) + self.gamma, 0, 1)
        return res

    def get_hard_value(self, x):
        init_shape = x.shape
        return ((torch.floor(x.reshape_as(self.alpha) / self.scale) + (self.alpha >= 0).float()) * self.scale).reshape(*init_shape)

    # def get_soft_targets(self):
    #     s = torch.sigmoid(self.beta * self.alpha)
    #     return s

    def update_weight(self):
        '''从这个方法里面对权重矩阵进行补偿，然后后面直接调用均匀量化器进行量化。'''


    def forward(self, weight):
        # print(f"AdaRoundGPTQQuantizer中的Hessian矩阵是否全0：{torch.equal(self.hessian,torch.zeros_like(self.hessian))}")
        # print(self.hessian)
        # print(f"调用forward方法时的weight.shape = {weight.shape}") # 已经进行拆分了
        # print(f"能否拿到index？{self.current_index}")
        H = self.hessian  # 确保 H 不参与梯度计算
        cols = self.cols
        inp = weight.clone()
        inp_origin = weight.clone()
        dead = torch.diag(H) == 0  # 找出对角线为0的位置，也就是不可逆或者无信息的特征方向
        H[dead, dead] = 1
        inp[:, dead] = 0  # 现在可以安全修改
        Losses = torch.zeros_like(inp)  # 统计损失
        inp_sim, inp_int = torch.zeros_like(inp), torch.zeros_like(inp)
        damp = self.percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(cols)
        H = H.clone()
        H[diag, diag] = H[diag,diag] + damp  # 对角线元素加上阻尼，正定矩阵一定是方阵。

        H = torch.linalg.cholesky(H)  # Cholesky分解，得到下三角矩阵
        H = torch.cholesky_inverse(H)  # 通过分解后的形式求逆矩阵，相比直接求逆矩阵，分解后的数值更加稳定
        H = torch.linalg.cholesky(H)  # 再对逆矩阵分解，得到上三角矩阵
        Hinv = H  # 改名，最后得到的是Hessian逆矩阵的Cholesky分解形式。
        # print(f"Hinv是否全0：{Hinv.count_nonzero() == 0}")
        for i1 in range(0, cols, self.blocksize):  # i1：0~columns，步长=blocksize
            i2 = min(i1 + self.blocksize, cols)
            count = i2 - i1  # 这是计算当前block有多少列，防止列数不足blocksize的时候超过范围
            # W1 = inp[:, i1:i2].clone()  # 分块之后的权重子矩阵
            W1 = inp[:, i1:i2]  # 分块之后的权重子矩阵
            Q1 = torch.zeros_like(W1)  # 子矩阵的量化结果
            Err1 = torch.zeros_like(W1)  # 用于存储当前block内每一列量化时产生的误差向量，当前block量化结束后要把误差传播到其他block
            Losses1 = torch.zeros_like(W1)  # 存储当前block内每一列量化带来的近似损失，估算当前block的总误差
            Hinv1 = Hinv[i1:i2, i1:i2]  # 权重子矩阵的Hessian矩阵对应的行和列
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                if self.round_mode == 'nearest':  # 四舍五入
                    q_int = torch.round(w.unsqueeze(1) / self.scale)
                elif self.round_mode == 'nearest_ste':  # 使用直通估计器的四舍五入
                    q_int = round_ste(w.unsqueeze(1) / self.scale)
                elif self.round_mode == 'learned_hard_sigmoid':  # 通过可学习的参数 alpha 决定舍入方向，自适应舍入
                    q_floor = torch.floor(w.unsqueeze(1) / self.scale)
                    if self.soft_targets:  # 训练时可用软目标提升梯度稳定性，推理时用硬判决保证效率。
                        q_int = q_floor + self.get_soft_targets(i).unsqueeze(1) # 这里不应该使用整个的维度，而是取出单独的列
                    else:
                        q_int = q_floor + (self.alpha >= 0).float()[:,i].unsqueeze(1)
                else:
                    raise ValueError('Wrong rounding mode')
                q_quant = torch.clamp(q_int + self.zero_point, 0, self.maxq)
                q_sim = self.scale * (q_quant - self.zero_point)
                q_sim = q_sim.flatten()

                Q1[:, i] = q_sim
                Losses1[:, i] = (w - q_sim) ** 2 / d ** 2
                err1 = (w - q_sim) / d
                '''2025年9月5日修正梯度无法传播的问题'''
                # W1[:, i:] = W1[:, i:] - err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                delta = err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))  # [C, count-i]
                delta_padded = F.pad(delta, (i, 0))
                W1 = W1 - delta_padded
                '''/2025年9月5日修正梯度无法传播的问题'''
                Err1[:, i] = err1
            inp_sim[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2
            # inp[:, i2:] = inp[:, i2:] - Err1.matmul(Hinv[i1:i2, i2:])
            '''2025年9月5日修正梯度无法传播的问题'''
            delta_tail = Err1.matmul(Hinv[i1:i2, i2:])  # [C, (cols - i2)]，只对应 i2..cols-1 列
            delta_tail_padded = F.pad(delta_tail, (i2, 0))  # 左侧补 i2 列 0 => [C, cols]
            inp = inp - delta_tail_padded
            '''/2025年9月5日修正梯度无法传播的问题'''
        # print(f"量化完成后的权重：{inp.shape}")
        # print(inp)
        # print(f"量化前后的inp是否一样：{torch.equal(inp, inp_origin)}") # 不应该啊，为什么量化前后的输入是一样的？按理说在量化的时候会对权重进行补偿，然后权重值就会发生变化？
        return inp_sim



    '''这是之前版本的forward方法，2025年9月1日我就把Hessian矩阵定义到了量化器里面，
    def forward(self, inp, H, col):
        print(f"H.shape {H.shape}，H={H}")  # 能拿到这一层的Hessian矩阵
        print(f"inp.shape {inp.shape}") # 对于第一个卷积层：shape=[192,768]
        print(f"self.scale.shape={self.scale.shape}")
        print(f"self.zero_point.shape={self.zero_point.shape}")
        inp = inp.clone()
        dead = torch.diag(H) == 0  # 找出对角线为0的位置，也就是不可逆或者无信息的特征方向
        H[dead, dead] = 1
        inp[:, dead] = 0
        Losses = torch.zeros_like(inp)  # 统计损失
        inp_sim, inp_int = torch.zeros_like(inp), torch.zeros_like(inp)
        damp = self.percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(col)
        H[diag, diag] += damp  # 对角线元素加上阻尼，正定矩阵一定是方阵。
        H = torch.linalg.cholesky(H)  # Cholesky分解，得到下三角矩阵
        H = torch.cholesky_inverse(H)  # 通过分解后的形式求逆矩阵，相比直接求逆矩阵，分解后的数值更加稳定
        H = torch.linalg.cholesky(H)  # 再对逆矩阵分解，得到上三角矩阵
        Hinv = H  # 改名，最后得到的是Hessian逆矩阵的Cholesky分解形式。
        for i1 in range(0, col, self.blocksize):  # i1：0~columns，步长=blocksize
            i2 = min(i1 + self.blocksize, col)
            count = i2 - i1  # 这是计算当前block有多少列，防止列数不足blocksize的时候超过范围
            W1 = inp[:, i1:i2].clone()  # 分块之后的权重子矩阵
            Q1 = torch.zeros_like(W1)  # 子矩阵的量化结果
            print(f"Q1.shape={Q1.shape}")
            print(f"W1.shape={W1.shape}")
            Err1 = torch.zeros_like(W1)  # 用于存储当前block内每一列量化时产生的误差向量，当前block量化结束后要把误差传播到其他block
            Losses1 = torch.zeros_like(W1)  # 存储当前block内每一列量化带来的近似损失，估算当前block的总误差
            Hinv1 = Hinv[i1:i2, i1:i2]  # 权重子矩阵的Hessian矩阵对应的行和列
            for i in range(count):
                w = W1[:, i]
                print(f"w.shape={w.shape}") # 192
                print(f"w.unsqueeze(1).shape={w.unsqueeze(1).shape}")
                d = Hinv1[i, i]
                # 舍入过程
                if self.round_mode == 'nearest':  # 四舍五入
                    q_int = torch.round(w.unsqueeze(1) / self.scale)
                    print(f"q_int.shape={q_int.shape}")  # shape=[192,768]
                elif self.round_mode == 'nearest_ste':  # 使用直通估计器的四舍五入
                    q_int = round_ste(w.unsqueeze(1) / self.scale)
                    print(f"q_int.shape={q_int.shape}")  # shape=[192,768]
                elif self.round_mode == 'learned_hard_sigmoid':  # 通过可学习的参数 alpha 决定舍入方向，自适应舍入
                    q_floor = torch.floor(w.unsqueeze(1) / self.scale)
                    print(f"q_floor.shape={q_floor.shape}")  # shape=[192,1]
                    if self.soft_targets:  # 训练时可用软目标提升梯度稳定性，推理时用硬判决保证效率。
                        q_int = q_floor + self.get_soft_targets()
                    else:
                        q_int = q_floor + (self.alpha >= 0).float()
                else:
                    raise ValueError('Wrong rounding mode')
                # 截断过程
                print(f"q_int.shape={q_int.shape}") # shape=[192,768]
                q_quant = torch.clamp(q_int + self.zero_point, 0, self.maxq)
                print(f"q_quant.shape={q_quant.shape}") # shape=[192,768]
                # 反量化
                q_sim = self.scale * (q_quant - self.zero_point)
                print(f"q_sim.shape={q_sim.shape}")
                q_sim = q_sim.flatten()
                print(f"q_sim.shape={q_sim.shape}")
                Q1[:, i] = q_sim
                Losses1[:, i] = (w - q_sim) ** 2 / d ** 2
                err1 = (w - q_sim) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1
            inp_sim[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2
            inp[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
        return inp_sim
    '''


    ''' 这是直接从修改的舍入模式，但是无法用到重建里面，因为重建的时候没有对矩阵单独处理。
    def forward(self, inp, H, col):
        print(f"H.shape {H.shape}，H={H}") # 能拿到这一层的Hessian矩阵
        print(f"inp.shape {inp.shape}")
        inp = inp.clone()
        dead = torch.diag(H) == 0  # 找出对角线为0的位置，也就是不可逆或者无信息的特征方向
        H[dead, dead] = 1
        inp[:, dead] = 0
        Losses = torch.zeros_like(inp)  # 统计损失
        inp_sim, inp_int = torch.zeros_like(inp), torch.zeros_like(inp)
        damp = self.percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(col)
        H[diag, diag] += damp  # 对角线元素加上阻尼，正定矩阵一定是方阵。
        H = torch.linalg.cholesky(H)  # Cholesky分解，得到下三角矩阵
        H = torch.cholesky_inverse(H)  # 通过分解后的形式求逆矩阵，相比直接求逆矩阵，分解后的数值更加稳定
        H = torch.linalg.cholesky(H)  # 再对逆矩阵分解，得到上三角矩阵
        Hinv = H  # 改名，最后得到的是Hessian逆矩阵的Cholesky分解形式。
        for i1 in range(0, col, self.blocksize):  # i1：0~columns，步长=blocksize
            i2 = min(i1 + self.blocksize, col)
            count = i2 - i1  # 这是计算当前block有多少列，防止列数不足blocksize的时候超过范围
            W1 = inp[:, i1:i2].clone()  # 分块之后的权重子矩阵
            Q1 = torch.zeros_like(W1)  # 子矩阵的量化结果
            Err1 = torch.zeros_like(W1)  # 用于存储当前block内每一列量化时产生的误差向量，当前block量化结束后要把误差传播到其他block
            Losses1 = torch.zeros_like(W1)  # 存储当前block内每一列量化带来的近似损失，估算当前block的总误差
            Hinv1 = Hinv[i1:i2, i1:i2]  # 权重子矩阵的Hessian矩阵对应的行和列
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                # 舍入过程
                if self.round_mode == 'nearest':  # 四舍五入
                    q_int = torch.round(w.unsqueeze(1) / self.scale)
                elif self.round_mode == 'nearest_ste':  # 使用直通估计器的四舍五入
                    q_int = round_ste(w.unsqueeze(1) / self.scale)
                elif self.round_mode == 'learned_hard_sigmoid':  # 通过可学习的参数 alpha 决定舍入方向，自适应舍入
                    q_floor = torch.floor(w.unsqueeze(1) / self.scale)
                    if self.soft_targets:  # 训练时可用软目标提升梯度稳定性，推理时用硬判决保证效率。
                        q_int = q_floor + self.get_soft_targets()
                    else:
                        q_int = q_floor + (self.alpha >= 0).float()
                else:
                    raise ValueError('Wrong rounding mode')
                # 截断过程
                q_quant = torch.clamp(q_int + self.zero_point, 0, self.maxq)
                # 反量化
                q_sim = self.scale * (q_quant - self.zero_point)
                q_sim = q_sim.flatten()
                Q1[:, i] = q_sim
                Losses1[:, i] = (w - q_sim) ** 2 / d ** 2
                err1 = (w - q_sim) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1
            inp_sim[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2
            inp[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
        return inp_sim
        '''
    def __repr__(self):
        return (f'{self.__class__.__name__}(n_bits={self.n_bits}, symmetric={self.symmetric}, '
                f'channel_wise={self.channel_wise}, round_mode={self.round_mode})')