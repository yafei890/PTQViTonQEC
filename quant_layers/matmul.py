import numpy as np
import math
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from itertools import product
from quantizers.uniform import UniformQuantizer

from quantizers.logarithm import Log2Quantizer, LogSqrt2Quantizer, AdaLogQuantizer
from datetime import datetime

'''矩阵乘法的量化层不能用GPTQ量化，那就简单了，直接把AdaLog的量化方法拿过来用。'''

class MinMaxQuantMatMul(nn.Module):
    '''好像量化算子的构建都是从Min-Max量化方法开始的，这是很简单的一个量化算子。'''

    def __init__(self,
                 A_bit=4,
                 B_bit=4,
                 mode="raw"):
        super().__init__()
        self.mode = mode
        self.A_quantizer = UniformQuantizer(n_bits=A_bit, symmetric=True, channel_wise=False)
        self.B_quantizer = UniformQuantizer(n_bits=B_bit, symmetric=True, channel_wise=False)
        self.raw_input = None
        self.raw_out = None
        self.tmp_input = None
        self.tmp_out = None
        self.calibrated = False

    def forward(self, A, B):
        if self.mode == 'raw' :
            out = A @ B
        elif self.mode == "quant_forward"  or self.mode == "quant_weight_only" :
            out = self.quant_forward(A, B)
        else:
            raise NotImplementedError
        return out

    def quant_input_A(self, x):
        return self.A_quantizer(x)

    def quant_input_B(self, x):
        return self.B_quantizer(x)

    def quant_forward(self, A, B):
        assert self.calibrated, f"Module should be calibrated before run quant_forward for {self}"
        '''校准是为了统计出量化参数scale和zp，没有这俩怎么量化？'''
        return self.quant_input_A(A) @ self.quant_input_B(B)


class PTQSLQuantMatMul(MinMaxQuantMatMul):
    """
    - Q @ K:
        - A's shape: B,H,S,C
        - B's shape: B,H,C,S
    - scores @ V:
        - A's shape: B,H,S,S
        - B's shape: B,H,S,C
    """

    def __init__(self,
                 A_bit=8,
                 B_bit=8,
                 mode="raw",
                 search_round=1,
                 eq_n=100,
                 head_channel_wise=True,  # 按注意力头量化，即每个注意力头有独立的量化参数
                 num_heads=12):
        super().__init__(A_bit, B_bit, mode)
        self.A_quantizer = UniformQuantizer(n_bits=A_bit, symmetric=True, channel_wise=head_channel_wise)
        self.B_quantizer = UniformQuantizer(n_bits=B_bit, symmetric=True, channel_wise=head_channel_wise)
        self.search_round = search_round
        self.eq_n = eq_n
        # the head dim is always dim-1
        self.head_channel_wise = head_channel_wise
        self.num_heads = num_heads
        '''是否按注意力头分配缩放因子，然后再将缩放因子设置为可学习的参数。运算的时候会利用广播机制对齐维度。'''
        target_shape = [1, self.num_heads, 1, 1] if self.head_channel_wise else [1, 1, 1, 1]
        self.A_quantizer.scale = nn.Parameter(torch.zeros(*target_shape))
        self.B_quantizer.scale = nn.Parameter(torch.zeros(*target_shape))

    def _get_similarity(self, tensor_raw, tensor_sim):
        '''用均方误差衡量量化的相似度'''
        return -(tensor_raw - tensor_sim) ** 2


class PTQSLBatchingQuantMatMul(PTQSLQuantMatMul):
    '''相比父类增加了批量校准'''

    def __init__(self,
                 A_bit=8,
                 B_bit=8,
                 mode="raw",
                 calib_batch_size=32,
                 search_round=1,
                 eq_n=100,
                 head_channel_wise=True,
                 num_heads=12):
        super().__init__(A_bit, B_bit, mode, search_round, eq_n, head_channel_wise, num_heads)
        self.calib_batch_size = calib_batch_size  # 指定每个校准批次的样本量，应该是32。

    def _initialize_calib_parameters(self):
        '''该方法动态计算并行处理的候选参数数量，确保不超过GPU内存限制。我感觉这个应该挺耗费时间的'''
        self.calib_size = self.raw_input[0].shape[0]  # self.raw_input是校准数据集（通常为tuple包含A/B矩阵）；取第一个矩阵的batch维度大小作为总校准样本量
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            memory = props.total_memory // 2  # 只使用一半的显存
        else:
            raise EnvironmentError("CUDA is not available on this system")
        '''统计元素的个数从而计算显存的占用
        输入数据：4字节/元素 * (A矩阵元素数 + B矩阵元素数)；输出数据：8字节/元素 * (输出矩阵元素数)（假设float16或double计算）'''
        numel = (4 * self.raw_input[0][:self.calib_size].numel() +
                 4 * self.raw_input[1][:self.calib_size].numel() +
                 8 * self.raw_out[:self.calib_batch_size].numel())  # number of parameters on GPU
        self.parallel_eq_n = int((memory / 4) // numel)  # 将总可用内存转换为float32元素数量（4字节/元素）；然后计算单批次能处理的候选参数数量
        self.parallel_eq_n = math.ceil(
            self.eq_n * 1.0 / math.ceil(self.eq_n * 1.0 / self.parallel_eq_n))  # 确保最终parallel_eq_n能整除eq_n
        # 尽可能大的并行度能加速搜索，但是太大有可能会爆显存


class AsymmetricallyBatchingQuantMatMul(PTQSLBatchingQuantMatMul):
    '''对应于matmul1，QK矩阵相乘'''

    def __init__(self,
                 A_bit=8,
                 B_bit=8,
                 mode="raw",
                 calib_batch_size=32,
                 search_round=1,
                 eq_n=128,
                 head_channel_wise=True,  # 按照分类头的个数划分通道数
                 num_heads=12,
                 fpcs=False,
                 steps=4):
        super().__init__(A_bit, B_bit, mode, calib_batch_size, search_round, eq_n, head_channel_wise, num_heads)
        self.fpcs = fpcs
        self.steps = steps

        del self.A_quantizer, self.B_quantizer  # 两个对称量化器都不要了，重新进行构建非对称量化器
        self.A_quantizer = UniformQuantizer(n_bits=A_bit, symmetric=False, channel_wise=head_channel_wise)
        self.B_quantizer = UniformQuantizer(n_bits=B_bit, symmetric=False, channel_wise=head_channel_wise)

        target_shape = [1, self.num_heads, 1, 1] if self.head_channel_wise else [1, 1, 1, 1]  # 根据分类头划分维度
        '''将矩阵A、B的缩放因子和零点设置成可学习的参数'''
        self.A_quantizer.scale = nn.Parameter(torch.zeros(*target_shape))
        self.B_quantizer.scale = nn.Parameter(torch.zeros(*target_shape))
        self.A_quantizer.zero_point = nn.Parameter(torch.zeros(*target_shape))
        self.B_quantizer.zero_point = nn.Parameter(torch.zeros(*target_shape))

    def _search_best_A_scale(self, A_scale_candidates, A_zero_point_candidates, topk=1):
        batch_similarities = []  # similarities, need to concatenate and calculate sum
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)  # 校准集的大小
            A = self.raw_input[0][b_st:b_ed].cuda()  # shape:[b, *, dim1, dim2]
            B = self.raw_input[1][
                b_st:b_ed].cuda()  # shape:[b, *, dim2, dim3] # 如果是A、B直接相乘，得到的维度是[b, *, dim1, dim3]，正好对应raw_out的维度
            '''这个方法是搜索A的最优缩放因子，因此直接使用已经量化的B。下面的_search_best_B_scale也是同样的道理'''
            B_sim = self.quant_input_B(B).unsqueeze(0)  # shape:[1,b,*,dim2,dim3]
            raw_out = self.raw_out[b_st:b_ed].unsqueeze(
                0).cuda()  # shape:[b, *, dim1, dim3]；这里的维度应该不是4维，而是5维：[1, b, *, dim1, dim3]，否则下面没法算相似度
            similarities = []
            for p_st in range(0, self.eq_n, self.parallel_eq_n):  # 并行搜索
                p_ed = min(self.eq_n, p_st + self.parallel_eq_n)
                '''quantize A：scale和zp的维度都是[parallel_eq_n, 1, 1, 1, 1](全局量化)或者[parallel_eq_n, 1, *, 1, 1](head_channel_wise=True)'''
                cur_A_scale = A_scale_candidates[p_st:p_ed]
                cur_A_zero_point = A_zero_point_candidates[p_st:p_ed]
                A_sim = A.squeeze(0)  # 去掉多余的维度，这里尝试删除A的第0维：如果第0维的维度大小是1，那么进行删除，否则不进行操作；A_sim's shape:[b, *, dim1, dim2]
                '''下面的运算要用到广播机制，扩展A_sim的维度'''
                A_quant = ((A_sim / cur_A_scale).round_() + cur_A_zero_point).clamp(0,
                                                                                    2 * self.A_quantizer.n_levels - 1)
                A_sim = (A_quant - cur_A_zero_point).mul_(cur_A_scale)  # shape: (parallel_eq_n,b,*,dim1,dim2)
                out_sim = A_sim @ B_sim  # shape: parallel_eq_n,b,*,dim1,dim3
                similarity = self._get_similarity(raw_out, out_sim)  # shape: parallel_eq_n,b,*,dim1,dim3
                if self.head_channel_wise:
                    '''这里对5维张量的后两个维度进行取均值压缩，只保留每个head的差异，得到一个3维的张量'''
                    similarity = torch.mean(similarity,
                                            dim=list(range(3, len(similarity.shape))))  # shape: parallel_eq_n,b,heads
                else:
                    '''如果head_channel_wise==False，那么similarity就是一个4维的张量，取均值后得到的就是一个全局的差异'''
                    similarity = torch.mean(similarity,
                                            dim=list(range(2, len(similarity.shape))))  # shape: parallel_eq_n,b
                '''batch内样本求和，且保留batch维度'''
                similarity = similarity.sum(dim=1, keepdim=True)  # shape: (parallel_eq_n,1) or (parallel_eq_n,1,heads)
                similarities.append(similarity)
            '''calculate best similarity for this block，拼起来并行的部分，parallel_eq_n->eq_n'''
            similarities = torch.cat(similarities, 0)  # shape: (eq_n,1) or (eq_n,1,heads)
            batch_similarities.append(similarities)
        '''合并掉batch的维度'''
        batch_similarities = torch.cat(batch_similarities, dim=1).sum(dim=1,
                                                                      keepdim=False)  # shape: eq_n or (eq_n,heads)
        _, best_index = torch.topk(batch_similarities, k=topk, dim=0)
        '''这里是为了匹配张量的形状，-1表示自动计算大小，1用来占位'''
        best_index = best_index.view(topk, 1, -1, 1, 1)  # 结果是一个5维的张量
        '''选择最优的量化参数'''
        if topk == 1:
            tmp_A_scale = torch.gather(A_scale_candidates, dim=0, index=best_index)
            tmp_A_zero_point = torch.gather(A_zero_point_candidates, dim=0, index=best_index)
            self.A_quantizer.scale.data.copy_(tmp_A_scale.view(self.A_quantizer.scale.shape))
            self.A_quantizer.zero_point.copy_(tmp_A_zero_point.view(self.A_quantizer.zero_point.shape))
        return best_index

    def _search_best_B_scale(self, B_scale_candidates, B_zero_point_candidates, topk=1):
        batch_similarities = []  # similarities, need to concatenate and calculate sum
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            A = self.raw_input[0][b_st:b_ed].cuda()
            B = self.raw_input[1][b_st:b_ed].cuda()
            A_sim = self.quant_input_A(A).unsqueeze(0)  # shape: 1,b,*,dim1,dim2
            raw_out = self.raw_out[b_st:b_ed].unsqueeze(0).cuda()
            similarities = []
            for p_st in range(0, self.eq_n, self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st + self.parallel_eq_n)
                # quantize B
                cur_B_scale = B_scale_candidates[p_st:p_ed]
                cur_B_zero_point = B_zero_point_candidates[p_st:p_ed]
                B_sim = B.squeeze(0)
                B_quant = ((B_sim / cur_B_scale).round_() + cur_B_zero_point).clamp(0,
                                                                                    2 * self.B_quantizer.n_levels - 1)
                B_sim = (B_quant - cur_B_zero_point).mul_(cur_B_scale)  # shape: (parallel_eq_n,b,*,dim2,dim3)
                out_sim = A_sim @ B_sim  # shape: parallel_eq_n,b,*,dim1,dim3
                similarity = self._get_similarity(raw_out, out_sim)  # shape: parallel_eq_n,b,*,dim1,dim3
                if self.head_channel_wise:
                    similarity = torch.mean(similarity,
                                            dim=list(range(3, len(similarity.shape))))  # shape: parallel_eq_n,b,heads
                else:
                    similarity = torch.mean(similarity,
                                            dim=list(range(2, len(similarity.shape))))  # shape: parallel_eq_n,b
                similarity = similarity.sum(dim=1, keepdim=True)  # shape: (parallel_eq_n,1) or (parallel_eq_n,1,heads)
                similarities.append(similarity)
            # calculate best similarity for this block
            similarities = torch.cat(similarities, 0)  # shape: (eq_n,1) or (eq_n,1,heads)
            batch_similarities.append(similarities)
        batch_similarities = torch.cat(batch_similarities, dim=1).sum(dim=1,
                                                                      keepdim=False)  # shape: eq_n or (eq_n,heads)
        _, best_index = torch.topk(batch_similarities, k=topk, dim=0)
        best_index = best_index.view(topk, 1, -1, 1, 1)
        if topk == 1:
            tmp_B_scale = torch.gather(B_scale_candidates, dim=0, index=best_index)
            tmp_B_zero_point = torch.gather(B_zero_point_candidates, dim=0, index=best_index)
            self.B_quantizer.scale.data.copy_(tmp_B_scale.view(self.B_quantizer.scale.shape))
            self.B_quantizer.zero_point.copy_(tmp_B_zero_point.view(self.B_quantizer.zero_point.shape))
        return best_index

    def calculate_percentile_candidates(self, x, l=0.9, r=1.0):
        num_zp = min(16, self.B_quantizer.n_levels)
        num_scale = int(self.eq_n / num_zp)
        percentiles_uppers, percentiles_lowers = [], []  # 初始化两个列表
        pct = torch.tensor([l, r])  # 计算百分位数：[0.9, 1.0]
        '''下面的两个参数用于动态调整计算方式，避免GPU内存不足'''
        tensor_too_large = True
        mini_batch_size = 1

        if self.head_channel_wise:  # 按分类头量化
            x_ = x.transpose(0, 1).contiguous()  # shape: heads,b,*,dim1,dim2；这个*表示中间可能有多个维度，只是没有展示出来——可变维度
            x_ = x_.view(x_.shape[0], mini_batch_size, -1)  # shape: heads,1,b×dim1×dim2/mini_batch_size
        else:  # 全局量化
            x_ = x.view(1, mini_batch_size, -1)
        while tensor_too_large:
            try:  # 尝试量化
                uppers_candidates = torch.quantile(x_, pct.to(x_.device), dim=-1).mean(dim=-1,
                                                                                       keepdim=False)  # shape: 2,(heads or 1)
                lowers_candidates = torch.quantile(x_, (1 - pct).to(x_.device), dim=-1).mean(dim=-1,
                                                                                             keepdim=False)  # shape: 2,(heads or 1)
                tensor_too_large = False
            except:  # 量化失败，增大mini_batch_size，为什么要这样做？
                '''首先，mini_batch_size=1的时候，x_的维度是heads,1,b×dim1×dim2/mini_batch_size，如果超过了GPU的显存大小，那么增大分母，就能减小显存的占用'''
                mini_batch_size *= 2  # 增大批次？
                x_ = x_.view(x_.shape[0], mini_batch_size, -1) if self.head_channel_wise else x_.view(1,
                                                                                                      mini_batch_size,
                                                                                                      -1)
        '''下面的两行语句计算的是中间的数值范围
        至于为什么这样计算动态范围，是因为作者采用了保守的计算方式。90~0忽略了极端大值的影响，100~10忽略了极端小值的影响。通过插值的方式生成候选范围（delta_min~delta_max），实现平衡。
        delta_min和delta_max的维度：1,1,num_heads,1,1。方便广播计算。
        下面的一系列操作我目前的理解是为了保留极端值（上下边界各10%的百分位数）的区间，直接舍弃掉肯定会丢失信息，这里让程序在后面的执行中自己决定是否保留极端值区间。
        现在还不是很理解这个操作，有点神奇。
        百分位数的含义：例如90%分位数，是指有90%的数小于这个数。0%分位数就是最小值，100%分位数就是最大值。
        delta_max - delta_min并不是整体数据范围的差值（并不是P100-P0的值，这个值表示的是全局的动态范围），而是宽松范围比保守范围多覆盖的区间，比如说：
        P0 = 0.1（最小值）、P10 = 0.2（10%分位数）、P90 = 1.8（90%分位数）P100 = 2.0（最大值）；
        P90 - P0 = 1.8 - 0.1 = 1.7；P100 - P10 = 2.0 - 0.2 = 1.8；delta_max - delta_min = 1.8 - 1.7 = 0.1；
        delta_max - delta_min衡量的是极端大值与极端小值范围的差异。
        '''
        delta_min = (uppers_candidates[0] - lowers_candidates[0]).view(1, 1, -1, 1,
                                                                       1)  # 数据的最小动态范围，90%-0%，忽略极端大值。其中-1是根据是否全局量化来决定
        delta_max = (uppers_candidates[1] - lowers_candidates[1]).view(1, 1, -1, 1,
                                                                       1)  # 数据的最大动态范围，100%-10%，忽略极端小值其中-1是根据是否全局量化来决定
        '''delta_max - delta_min的含义：P100-P10-(P90-P0)=P100-P10-P90+P0=(P100-P90)+(P0-P10)'''
        splits = torch.linspace(0, 1, steps=num_scale).cuda()[:, None, None, None, None] * (delta_max - delta_min)  # 插值
        '''优先覆盖主要数据分布（delta_min），再逐步试探极端值的影响。下面的这条语句会生成从保守到宽松的范围，也就是从delta_min到delta_max的宽松范围。
        然后在下面的超参数搜索中，候选值选用的是scale_candidates[-2]维度的候选值，因为这个维度是接近宽松范围但略保守的值，平衡了动态范围和稳定性。'''
        scale_candidates = (delta_min + splits).repeat(num_zp, 1, 1, 1, 1) / (
                    2 * self.B_quantizer.n_levels - 1)  # 插值后生成候选值

        '''以理论零点（n_levels应该是量化的级别，8bit下是256级）为中心，左右对称扩展 num_zp/2。'''
        zp_min = int(self.B_quantizer.n_levels - num_zp / 2)
        zp_max = int(self.B_quantizer.n_levels + num_zp / 2)
        '''range(zp_min, zp_max)一共是num_zp个点'''
        zp_candidates = torch.tensor(range(zp_min, zp_max)).cuda()  # 生成区间[zp_min, zp_max) 内所有整数候选的一维张量
        '''与scale的维度对齐，每个zp都要重复num_scale次，以保证量化参数匹配，并扩展成5维的张量
        也就是计算scale和zp的笛卡尔积：num_scale*num_zp'''
        zero_point_candidates = zp_candidates.repeat_interleave(num_scale)[:, None, None, None, None]
        zero_point_candidates = zero_point_candidates.repeat(1,
                                                             *scale_candidates.shape[1:])  # 扩展维度，与scale_candidates完全一致。
        return scale_candidates, zero_point_candidates  # 返回的还是缩放因子和零点的候选值

    def _fpcs(self, x, fpcs_width=16, steps=6, search_strategy=None):
        '''
        fpcs_width是每轮保留的候选参数数量，默认16个
        steps是迭代搜索的次数
        search_strategy决定搜哪个，比如_search_best_A_scale、_search_best_B_scale，需要在调用此方法时从外部给出。
        '''
        fpcs_new_cnt = int(self.eq_n / fpcs_width)  # 计算每轮细分后的新候选数
        scale_candidates, zero_point_candidates = self.calculate_percentile_candidates(x)  # 基于百分位数生成初始的scale和zp候选值
        delta_scale = scale_candidates[1:2] - scale_candidates[0:1]  # 还是获取一个scale的动态范围？
        '''下面的这条语句得到的topk_scale_candidates和topk_zp_candidates共享相同的下标，因为它们需要保持参数组合的配对关系'''
        topk_index = search_strategy(self, scale_candidates, zero_point_candidates, topk=fpcs_width)
        '''所以能通过一个下标分别拿到topk_zp_candidates和topk_scale_candidates '''
        topk_scale_candidates = torch.gather(scale_candidates, dim=0, index=topk_index)  # 拿到最优的前k个scale
        topk_zp_candidates = torch.gather(zero_point_candidates, dim=0, index=topk_index)  # 拿到最优的前k个zp
        remain_steps = steps - 1
        while remain_steps > 0:
            '''关于delta_scale 的理解：delta_scale 表示当前轮次 scale 的 搜索半径，即候选 scale 可以在当前最优值附近波动的最大范围。
            每轮迭代后，delta_scale 会按一定比例缩小，使得搜索范围逐步收窄。在每轮迭代中，delta_scale 用于生成 scale 的候选偏移量。
            linespace的作用是生成均匀的偏移量，然后把这个偏移量作用到delta_scale上，使得能够在delta_scale附近偏移。'''
            delta_scale_candidates = (torch.linspace(0, 1, steps=fpcs_new_cnt).cuda()[:, None, None, None,
                                      None] - 0.5) * delta_scale
            delta_scale = delta_scale / (fpcs_new_cnt - 0.5)  # 逐渐减小波动范围，实现精细化的搜索
            '''在最优的几个量化参数候选值附近进行偏移，然后扩展搜索。'''
            scale_candidates = (topk_scale_candidates.unsqueeze(1) + delta_scale_candidates.unsqueeze(0)).reshape(-1,
                                                                                                                  *scale_candidates.shape[
                                                                                                                   1:])
            zero_point_candidates = topk_zp_candidates.repeat_interleave(fpcs_new_cnt, dim=0)
            topk_index = search_strategy(self, scale_candidates, zero_point_candidates,
                                         topk=1 if remain_steps == 1 else fpcs_width)
            if remain_steps > 1:  # 需要接着搜索，拿到本次搜索出的候选值。
                topk_scale_candidates = torch.gather(scale_candidates, dim=0, index=topk_index)
                topk_zp_candidates = torch.gather(zero_point_candidates, dim=0, index=topk_index)
            remain_steps -= 1

    def params_searching(self):  # 主调用方法，仅仅是找到量化参数，那在哪里进行了量化呢？
        '''
        本方法是本类的核心方法，负责搜索并确定最优的量化参数（scale 和 zero_point），以最小化量化误差。
        它结合了粗搜索（基于百分位数）和细搜索（FPCS 或直接搜索）两种策略，确保量化后的矩阵乘法结果尽可能接近原始浮点计算。
        '''
        self._initialize_calib_parameters()  # 初始化校准参数，主要初始化的是parallel_eq_n，防止在量化的时候爆显存
        '''基于百分位数生成A和B的量化参数候选值，粗搜索'''
        A_scale_candidates, A_zero_point_candidates = self.calculate_percentile_candidates(self.raw_input[0].cuda())
        B_scale_candidates, B_zero_point_candidates = self.calculate_percentile_candidates(self.raw_input[1].cuda())
        '''下面的语句才是设置初始量化参数：初始化量化器的 scale 和 zp，避免随机值导致搜索不稳定
        A_scale_candidates[-2] 是倒数第二个候选值，通常对应较宽松的动态范围（P100-P10），作为搜索起点更鲁棒。'''
        self.A_quantizer.scale.data.copy_(A_scale_candidates[-2])
        self.A_quantizer.zero_point.data.copy_(A_zero_point_candidates[-2])
        self.B_quantizer.scale.data.copy_(B_scale_candidates[-2])
        self.B_quantizer.zero_point.data.copy_(B_zero_point_candidates[-2])
        self.A_quantizer.inited, self.B_quantizer.inited = True, True  # 标记为已经初始化

        for e in range(self.search_round):  # 多轮搜索
            if self.fpcs:  # 启用渐进式搜索
                self._fpcs(self.raw_input[0].cuda(), steps=self.steps,
                           search_strategy=AsymmetricallyBatchingQuantMatMul._search_best_A_scale)
                self._fpcs(self.raw_input[1].cuda(), steps=self.steps,
                           search_strategy=AsymmetricallyBatchingQuantMatMul._search_best_B_scale)
            else:  # 不启用渐进式搜索
                self._search_best_A_scale(A_scale_candidates, A_zero_point_candidates)
                self._search_best_B_scale(B_scale_candidates, B_zero_point_candidates)
        self.calibrated = True  # 标记为已校准
        del self.raw_input, self.raw_out
        return None


class PostSoftmaxAsymmetricallyBatchingQuantMatMul(AsymmetricallyBatchingQuantMatMul):
    '''
    对应于matmul2，softmax和V矩阵相乘。
    对softmax之后的矩阵乘进行量化，同样是矩阵乘法中的两个因子，矩阵A和矩阵B，对他们的处理是不一样的。
    A矩阵对应于注意力机制中softmax之后的得分矩阵，而B矩阵对应于V矩阵，B矩阵在之前不需要经过softmax操作，因此不需要A矩阵那样额外的处理。
    log_2 base = 1/k
    k = r / q, here we set r = 37, and search the best q.
    r 是固定值，q 是可搜索参数，通过搜索 q 来间接调整底数，避免直接优化底数导致数值不稳定。
    将底数拆分乘两个数相除的形式，固定其中的分子，搜索分母。
    '''

    def __init__(self,
                 A_bit=8,
                 B_bit=8,
                 mode="raw",
                 calib_batch_size=32,
                 search_round=1,
                 eq_n=100,
                 head_channel_wise=True,
                 num_heads=12,
                 fpcs=False,
                 steps=4,
                 quantizer='adalog'  # 这里采用adalog量化器
                 ):
        super().__init__(A_bit, B_bit, mode, calib_batch_size, search_round,
                         eq_n, head_channel_wise, num_heads, fpcs, steps)
        del self.A_quantizer  # 要重新构建量化器，不是原来的均匀量化器，但是保留B的均匀量化器。这就是对softmax激活值重新构建激活值量化器，然后对V还是使用均匀量化。
        target_shape = [1, 1, 1, 1]  # 目标张量的维度？
        if quantizer == 'log2':
            self.A_quantizer = Log2Quantizer(n_bits=A_bit, symmetric=False, channel_wise=False)
        elif quantizer == 'logsqrt2':
            self.A_quantizer = LogSqrt2Quantizer(n_bits=A_bit, symmetric=False, channel_wise=False)
        elif quantizer == 'adalog':
            self.A_quantizer = AdaLogQuantizer(n_bits=A_bit, symmetric=False, channel_wise=False)
            self.table = torch.tensor([2 ** (-j / self.A_quantizer.r) for j in range(120)])
            self.table_scale = 1. / (4 * self.A_quantizer.n_levels - 2)
            self.table = torch.round(self.table / self.table_scale) * self.table_scale  # 量化之后再反量化
        else:
            raise NotImplementedError(f"quantizer {quantizer} not implemented!")
        self.A_quantizer.scale = nn.Parameter(torch.ones(target_shape))  # 将scale设置成可学习的参数
        self.A_quantizer.inited = True

    def _search_best_A_log_base(self, q_candidates=None, topk=1):
        if q_candidates is None:  # 生成初始的候选参数
            q_candidates = torch.tensor([i for i in range(10, 11 + self.eq_n)]).cuda().view(-1, 1, 1, 1,
                                                                                            1)  # shape:en_n,1,1,1,1
        batch_similarities = []  # similarities, need to concatenate and calculate sum
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            A = self.raw_input[0][b_st:b_ed].cuda()
            B = self.raw_input[1][b_st:b_ed].cuda()
            B_sim = self.quant_input_B(B).unsqueeze(0)  # shape: 1,b,*,dim2,dim3
            raw_out = self.raw_out[b_st:b_ed].unsqueeze(0).cuda()
            similarities = []
            for p_st in range(0, self.eq_n, self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st + self.parallel_eq_n)
                # quantize A
                cur_q = q_candidates[p_st:p_ed]
                A_sim = A.squeeze(0)
                A_sim_quant = torch.round(
                    -A_sim.log2() * self.A_quantizer.r / cur_q)  # shape: parallel_eq_n, b, heads, *
                mask = A_sim_quant >= 2 * self.A_quantizer.n_levels  # mask用于标记溢出的位置
                A_sim_quant = A_sim_quant.clamp_(0, 2 * self.A_quantizer.n_levels - 1)
                '''col_index是用来查表的操作，remainder是取余'''
                col_index = torch.remainder(A_sim_quant * cur_q, self.A_quantizer.r).round_().long()
                '''反量化，将计算的过程替换为了查表。之前好像还需要判断是训练模式还是推理模式，两个模式有不同的处理方式，这里直接不需要判断了'''
                A_sim = (2 ** (-1 * torch.floor(A_sim_quant * cur_q / self.A_quantizer.r))) * self.table.to(A.device)[
                    col_index]
                A_sim[mask] = 0  # 溢出的位置置0
                '''计算量化后的结果'''
                out_sim = A_sim @ B_sim  # shape: parallel_eq_n,b,*,dim1,dim3
                '''然后就是计算相似度，下面的这些代码好像跟AsymmetricallyBatchingQuantMatMul类中的_search_best_A_log_base几乎一模一样。
                唯一的差别是AsymmetricallyBatchingQuantMatMul类中要看是否启用分头量化，这里不需要考虑这个问题。'''
                similarity = self._get_similarity(raw_out, out_sim)  # shape: parallel_eq_n,b,*,dim1,dim3
                similarity = torch.mean(similarity, dim=list(range(2, len(similarity.shape))))  # shape: parallel_eq_n,b
                similarity = similarity.sum(dim=1, keepdim=True)  # shape: (parallel_eq_n,1) or (parallel_eq_n,1,heads)
                similarities.append(similarity)
            # calculate best similarity for this block
            similarities = torch.cat(similarities, 0)  # shape: (eq_n,1) or (eq_n,1,heads)
            batch_similarities.append(similarities)
        batch_similarities = torch.cat(batch_similarities, dim=1).sum(dim=1, keepdim=True)  # shape: eq_n,1
        _, best_index = torch.topk(batch_similarities, k=topk, dim=0)
        best_index = best_index.view(topk, 1, 1, 1, 1)  # shape: topk,...
        if topk == 1:
            tmp_q = torch.gather(q_candidates, dim=0, index=best_index)  # 提取最优的q值
            self.A_quantizer.q.data.copy_(tmp_q.view(*self.A_quantizer.q.shape))  # 更新量化器
            self.A_quantizer.update_table()  # 再更新量化表
        return best_index

    def params_searching(self):
        self._initialize_calib_parameters()
        '''对B矩阵只需要按照之前的处理方法处理即可，也就是使用均匀量化器'''
        B_scale_candidates, B_zero_point_candidates = self.calculate_percentile_candidates(self.raw_input[1].cuda())
        self.B_quantizer.scale.data.copy_(B_scale_candidates[-2])
        self.B_quantizer.zero_point.data.copy_(B_zero_point_candidates[-2])
        self.B_quantizer.inited = True
        '''对矩阵A使用AdaLog量化器，搜索超参数。'''
        for e in range(self.search_round):
            if isinstance(self.A_quantizer, AdaLogQuantizer):  # 矩阵A必须使用AdaLog
                self._search_best_A_log_base()  # 这里搜索矩阵A量化器的底数，不需要使用FPCS，那是搜索scale和zp的。

            if self.fpcs:  # 使用FPCS搜索矩阵B的量化参数scale和zp，
                self._fpcs(self.raw_input[1].cuda(), steps=self.steps,
                           search_strategy=AsymmetricallyBatchingQuantMatMul._search_best_B_scale)
            else:  # 不使用FPCS，直接搜索：从初始参数的直接选择最优解
                self._search_best_B_scale(B_scale_candidates, B_zero_point_candidates)
            if not isinstance(self.A_quantizer, AdaLogQuantizer):  # 如果不使用AdaLog直接退出循环
                break
        self.calibrated = True
        del self.raw_input, self.raw_out
        return None


    # 63.956，63.931，64.976，64.848，64.593 AVG=64.466
