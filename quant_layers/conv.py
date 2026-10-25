import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
import sys

from mpmath import zeros

from quantizers.uniform import UniformQuantizer
from quantizers.gptq import  GPTQQuantizer
import math
import time

'''
如何构建？
Yuan和Zhu构建量化层主要是为了搜索量化参数，然后进行量化。我这里不需要很复杂的搜参数过程，而且能将GPTQ中对每一层的量化代码拿过来。
'''

# class GPTQConv2DWeight(nn.Conv2D):
#     '''
#     这个类的目的是重新封装quant.py中的量化器Quantizer
#     '''
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  kernel_size,
#                  stride=1,
#                  padding=0,
#                  dilation=1,
#                  groups: int = 1,
#                  bias: bool = True,
#                  padding_mode: str = 'zeros',
#                  mode='raw',
#                  w_bit=4,
#
#                  ):
#         super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
#         # self.mode = mode
#         self.w_bit = w_bit
#         # assert bias_bit is None, "No support bias bit now"
#         self.raw_input = None
#         self.raw_out = None
#         self.w_qmax = 2 ** (self.w_bit - 1)
#         self.w_quantizer = Quantizer(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
#
#
#         '''好像不需要单独定义量化参数？是的，量化参数是在gptq算法中定义的。'''
#
#         self.w_scale = None
#         self.w_zero_point = None
#
#
#     def quant_weight_bias(self):
#         W = layer.weight.data.clone()  # 获取权重数据
#         # w=(self.weight/self.w_scale).round_().clamp_(-self.w_qmax,self.w_qmax-1)
#         # w_sim=w.mul_(self.w_scale)
#         # if self.bias is not None:
#         #     return w_sim,self.bias
#         #     # bias=(self.bias/self.bias_interval).round_().clamp_(-self.bias_qmax,self.bias_qmax-1)
#         #     # bias_sim=bias*self.bias_interval
#         #     # return w_sim,bias_sim
#         # else:
#         #     return w_sim,None
#
#     def quant_input(self,x):
#         '''要在外面给a_scale赋值'''
#         x_sim=(x/self.a_scale).round_().clamp_(-self.a_qmax,self.a_qmax-1)
#         x_sim.mul_(self.a_scale)
#         return x_sim
#
#     def quant_forward(self,x):
#         # assert self.calibrated is not None,f"You should run calibrate_forward before run quant_forward for {self}"
#         w_sim,bias_sim=self.quant_weight_bias() # 这里要改，权重要用GPTQ量化
#         x_sim=self.quant_input(x)
#         out=F.conv2d(x_sim, w_sim, bias_sim, self.stride, self.padding, self.dilation, self.groups)
#         return out
#
#     def forward(self, x):
#         out = self.quant_forward(x)
#         return out
class MinMaxQuantConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 mode='raw',
                 w_bit=4,
                 a_bit=4):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.mode = mode
        self.w_quantizer = UniformQuantizer(n_bits=w_bit, symmetric=False, channel_wise=True)  # 权重和激活值均使用均匀量化器
        self.a_quantizer = UniformQuantizer(n_bits=a_bit, symmetric=True, channel_wise=False) # 输入值应该使用非对称量化器，因为图像是[0.255]
        '''但是AdaLog也是使用的对称量化？'''
        self.raw_input = None
        self.raw_out = None
        self.tmp_input = None
        self.tmp_out = None
        self.calibrated = False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.quantized_weight = torch.zeros(self.weight.shape)  # 创建一个新的张量，用于保存量化后的权重
        self.register_buffer('quantized_weight', self.weight.detach().clone())
        # self.quantized_weight.to(self.device)
        # self.quantized_weight = nn.Parameter(torch.ones_like(self.weight))

    def forward(self, x):
        if self.mode == 'raw':  # 不量化
            out = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        elif self.mode == "quant_forward" :
            out = self.quant_forward(x)
        elif self.mode == 'debug_only_quant_act':
            out = self.debug_only_quant_act(x)
        elif self.mode == "quant_weight_only":
            out = self.quant_weight_only(x)
        else:
            raise NotImplementedError
        return out

    def quant_weight_only(self, x):
        w_sim, bias_sim = self.quant_weight_bias()
        out = F.conv2d(x,w_sim,bias_sim,self.stride,self.padding,self.dilation,self.groups)
        return out
    # def quant_weight_bias(self):
    #     w_sim = self.w_quantizer(self.weight) # 需要让量化器返回反量化后的值
    #     return w_sim, self.bias if self.bias is not None else None

    def quant_input(self, x):
        if self.a_quantizer.n_bits >= 8:  # a_bit >= 8（如 8/16/32bit），视为全精度，直接返回 x（不量化）
            # print(f"卷积层的a_bit = {self.a_quantizer.n_bits}")
            return x
        return self.a_quantizer(x)

    def quant_forward(self, x):
        assert self.calibrated, f"Module should be calibrated before run quant_forward for {self}"  # 量化前应完成校准
        w_sim, bias_sim = self.quant_weight_bias() # 量化模式不需要在对权重和偏差进行量化了，因为calibrated之后这里保存的数据已经是反量化之后的权重数据了
        x_sim = self.quant_input(x) # 只需要对输入值进行均匀量化即可；试一下如果不量化权重结果怎么样
        out = F.conv2d(x_sim,w_sim,bias_sim,self.stride,self.padding,self.dilation,self.groups)
        return out
    def debug_only_quant_act(self, x):
        '''卷积层的输入值/激活值不是不用量化吗？'''
        x_sim = self.quant_input(x)
        out = F.conv2d(x_sim, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out

    # def quant_weight_bias(self):
    #     '''先留着，万一要用呢。'''
    #     # self.weight_scale shape: (1, 1) or (oc, 1)
    #     # self.weight       shape: (oc,ic,kw,kh)
    #     oc, ic, kw, kh = self.weight.data.shape
    #
    #     '''先调用w_quantizer逐通道量化（每个oc独立计算scale）；然后恢复权重形状为 (oc, ic, kw, kh) 并返回'''
    #     w_sim = self.w_quantizer(self.weight.view(oc, ic * kw * kh)).view(oc, ic, kw, kh)
    #     return w_sim, self.bias if self.bias is not None else None


# class GPTQConv2D(nn.Conv2d):
class GPTQConv2D(MinMaxQuantConv2d):
    '''再次明确一下，这个类在校准过程中的任务是确定权重的量化与反量化，并搜索激活值均匀量化器的量化参数。'''
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups: int = 1,
                 bias: bool = False,
                 padding_mode: str = 'zeros',
                 channel_wise: bool = False,
                 mse : bool = True,
                 symmetric : bool = False,
                 w_bit=4,
                 a_bit=8,
                 blocksize=128,  # 论文中的block
                 percdamp=.01,  # 添加到 Hessian 对角线的阻尼比例（防止奇异）
                 groupsize=-1,  # 对权重列进行分组以共享量化参数，-1 表示不分组
                 actorder=False,  # 是否按照激活重要性对列重新排序（论文提到的 activation-ordering）
                 static_groups=False,  # 是否对组的量化器静态缓存
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 mode = 'raw',
                 eq_n=128,
                 calib_batch_size = 32,
                 search_round = 3,
                 fpcs = False,
                 steps = 6,
                 ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode,
                         mode,w_bit, a_bit )
        self.w_bit = w_bit
        # self.a_bit = a_bit
        self.blocksize = blocksize
        self.percdamp = percdamp
        self.groupsize = groupsize
        self.actorder = actorder
        self.static_groups = static_groups
        self.channel_wise = channel_wise
        self.mse = mse
        self.symmetric = symmetric
        del self.w_quantizer # 针对权重重新构建GPTQ量化器，输入值还是保留原来的均匀量化
        self.w_quantizer = GPTQQuantizer(n_bits=self.w_bit,channel_wise=self.channel_wise,mse=self.mse,symmetric=self.symmetric)
        # 如果使用GPTQ自带的find_params方法搜索量化参数，就要把下面两行注释掉。如果使用AdaLog中的搜索方法，则保留。
        '''2025年8月22日：权重量化参数在初始化的时候还是2维的，为什么搜索完量化参数就变成4维的了？'''
        self.w_quantizer.scale = nn.Parameter(torch.zeros(out_channels,1)) # 通道级缩放因子
        self.w_quantizer.zero_point = nn.Parameter(torch.zeros(out_channels,1))  # 通道级的零点
        self.device = device
        self.mode = mode
        self.calib_batch_size = calib_batch_size
        self.eq_n = eq_n
        self.a_quantizer.scale = nn.Parameter(torch.zeros((1, 1, 1, 1)))
        # self.quantized_weight = torch.zeros(self.weight.shape) # 创建一个新的张量，用于保存量化后的权重
        # self.w_quantizer
        # self.nsamples = 0
        '''FPCS'''
        self.fpcs = fpcs
        self.search_round = search_round
        self.steps = steps

        '''创建Hessian矩阵'''
        # W = self.weight.data.clone()
        # W = W.flatten(1)
        # self.rows = W.shape[0]  # 行数,192
        # self.columns = W.shape[1]  # 列数,768
        # self.H = torch.zeros(self.columns, self.columns).to(self.device) # 这个Hessian矩阵是卷积层的Hessian矩阵，而不是GPTQ量化器的
        '''暂存输入和输出'''
        self.raw_input = None
        self.raw_out = None
        self.tmp_input = None
        self.tmp_out = None
        self.calibrated = False
        '''2025年8月22日：GPTQQuantizer参数搜索需要的属性：'''
        self.maxshrink = 0.8
        self.grid = 100
        self.norm = 2.4


    def cosine_similarity(self, tensor1, tensor2, eps=1e-8):
        tensor1_flat = tensor1.flatten()
        tensor2_flat = tensor2.flatten()
        return F.cosine_similarity(tensor1_flat.unsqueeze(0), tensor2_flat.unsqueeze(0), dim=1, eps=eps).item()
    """2025年8月22日：分理参数搜索和量化"""
    def quant_weight_bias(self):
        w = self.weight.data.clone()
        oc, ic, kw, kh = self.weight.data.shape
        # print(f"卷积层调用量化之前权重的维度：self.weight.data.shape = {self.weight.data.shape}")
        w_sim_t = self.w_quantizer(w.view(oc, ic * kw * kh)) # 传入量化器的权重变成了[192,768]的张量。
        w_sim = w_sim_t.view(oc, ic, kw, kh) # 恢复到原来的维度
        # print(f"权重量化前后的余弦相似度：{self.cosine_similarity(w_sim,w)}")
        return w_sim, self.bias if self.bias is not None else None

    def calculate_Hessian(self):
        '''针对卷积层来说，输入数据是[B,C,H,W]'''
        inp = self.raw_input.to(self.device).clone() # 32,3,224,224
        weight = self.weight.data.clone()
        weight = weight.flatten(1) # 192,768
        self.w_quantizer.rows = weight.shape[0]
        self.w_quantizer.cols = weight.shape[1]
        self.w_quantizer.hessian = torch.zeros(self.w_quantizer.cols, self.w_quantizer.cols).to(self.device)
        tmp = inp.shape[0]  # 拿到批次数据
        if isinstance(self, nn.Conv2d):
            '''二维卷积层，将输入的patch提取成矩阵，nn.Unfold会把卷积滑窗区域展开成列向量'''
            unfold = nn.Unfold(
                self.kernel_size,
                dilation=self.dilation,
                padding=self.padding,
                stride=self.stride
            )
            inp = unfold(inp)  # 这里是将inp展平，2×2的卷积核展成一列，不涉及运算，有点类似im2col；shape:[B, K*KH*KW, L]，L是分成多少个块，K=C_in*KH*KW是每个patch的输入维度。
            inp = inp.permute([1, 0, 2])  # 交换维度？shape:[K,B,L]
            inp = inp.flatten(1)  # 展平 shape[K,B×L]
        self.w_quantizer.hessian *= self.nsamples / (self.nsamples + tmp)  # 滑动平均
        self.nsamples += tmp  # 更新总样本数
        inp = math.sqrt(2 / self.nsamples) * inp.float()  # 归一化缩放
        self.w_quantizer.hessian += inp.matmul(inp.t())  # 对应论文中H=XX^T。 这里乘法用的就是数学意义上的矩阵乘，但依然能实现H的计算。



    """===========================通过百分位数计算输入值量化参数 2025年8月13日16点18分======================================"""
    # 看来我早就想这么干了，但是一直没干成
    def _get_similarity(self, tensor_raw, tensor_sim):
        return -(tensor_raw - tensor_sim) ** 2 # 计算原始张量和量化后张量的负均方误差。
    def _initialize_calib_parameters(self):
        self.calib_size = self.raw_input.shape[0]  # 记录样本的数量
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)  # 获取GPU的属性
            memory = props.total_memory // 2  # 取 GPU 总内存的一半作为安全阈值（避免内存溢出），这里的单位是字节？
        else:
            raise EnvironmentError("CUDA is not available on this system")
        numel = 2 * self.raw_input[:self.calib_batch_size].numel() + 2 * self.raw_out[
                                                                         :self.calib_batch_size].numel()  # number of parameters on GPU
        self.parallel_eq_n = int((memory / 4) // numel)  # 根据GPU内存和单批次数据量，计算理论最大并行候选数。memory / 4 计算的是字节数
        self.parallel_eq_n = math.ceil(self.eq_n * 1.0 / math.ceil(self.eq_n * 1.0 / self.parallel_eq_n))

    def calculate_percentile_weight_candidates(self, l=0.9, r=1.0):
        num_zp = self.w_quantizer.n_levels  # 量化级别的数量
        num_scale = int(self.eq_n / num_zp)  # 这个是保证候选值的数量是恒定的eq_n=num_scale*num_zp
        pct = torch.tensor([l, r])  # 百分位数
        w_uppers_candidates = torch.quantile(  # 上界的候选值
            self.weight.view(self.out_channels, -1),  # 重塑权重张量，形状变为：(oc,ic*kw*kh)
            pct.to(self.weight.device),  # 将分位数转移到GPU，确保分位点张量与权重在同一设备
            dim=-1  # 表示在展平后的最后一个维度（即展平后的空间维度）计算分位数；对形状 (out_channels, N) 的矩阵，会在每行的N个元素中计算分位数
        ).unsqueeze(-1)  # shape: 2, out_channels, 1 # 在结果张量的最后添加一个维度（从 [out_channels] 变为 [out_channels, 1]），这是确保每个通道有独立的分位数
        w_lowers_candidates = torch.quantile(  # 下界的候选值
            self.weight.view(self.out_channels, -1),
            (1 - pct).to(self.weight.device),
            dim=-1
        ).unsqueeze(-1)  # shapeL 2, out_channels, 1 # 2对应两个分位数
        delta_min = w_uppers_candidates[0:1] - w_lowers_candidates[0:1]  # 90%分位数的上界-10%分位数的下界，即原区间80%的中间范围的宽度，也就是一个“主体范围”的宽度
        delta_max = w_uppers_candidates[1:] - w_lowers_candidates[1:]  # 最大值-最小值，也就是两个边界值的差，“全局范围”的宽度
        splits = torch.linspace(0, 1, steps=num_scale).cuda()[:, None, None] * (delta_max - delta_min)
        weight_scale_candidates = (delta_min + splits).repeat(num_zp, 1, 1) / (2 * self.w_quantizer.n_levels - 1)
        zp_min = int(self.w_quantizer.n_levels - num_zp / 2)
        zp_max = int(self.w_quantizer.n_levels + num_zp / 2)
        zp_candidates = torch.tensor(range(zp_min, zp_max)).cuda()  # 创建候选序列
        weight_zero_point_candidates = zp_candidates.repeat_interleave(num_scale)[:, None, None]  # 确保零点与scale的数量对齐，shape:(eq_n,1,1)
        weight_zero_point_candidates = weight_zero_point_candidates.repeat(1, self.out_channels, 1)  # shape:(eq_n,out_channel,1)
        return weight_scale_candidates, weight_zero_point_candidates

    def _search_best_w_scale(self, weight_scale_candidates, weight_zero_point_candidates, topk=1):
        batch_similarities = []
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            x = self.raw_input[b_st:b_ed].cuda()
            raw_out = self.raw_out[b_st:b_ed].cuda().unsqueeze(1)  # shape: b,1,oc,fw,fh
            similarities = []
            for p_st in range(0, self.eq_n, self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st + self.parallel_eq_n)
                cur_w_scale = weight_scale_candidates[p_st:p_ed]  # shape: (parallel_eq_n, 1, 1) or (parallel_eq_n, oc, 1)
                cur_w_zero_point = weight_zero_point_candidates[p_st:p_ed]

                # quantize weight and bias 处理权重和偏差
                oc, ic, kw, kh = self.weight.data.shape
                w_sim = self.weight.view(oc, -1).unsqueeze(0)  # shape: (1, oc, ic*kw*kh) # 将权重的后三个维度合并，然后在第一维再添加一个维度
                w_quant = ((w_sim / cur_w_scale).round_() + cur_w_zero_point).clamp(0, 2 * self.w_quantizer.n_levels - 1)  # 非对称量化公式
                w_sim = (w_quant - cur_w_zero_point).mul_(cur_w_scale)  # shape: (parallel_eq_n,oc,ic*kw*kh) # 反量化
                w_sim = w_sim.view(-1, ic, kw, kh)  # shape: parallel_eq_n*oc,ic,kw,kh # 展开张量
                bias_sim = self.bias.repeat(p_ed - p_st) if self.bias is not None else None

                # quantize input # 处理激活值，此类中激活值仍采用均匀量化，直接调用方法就行了
                x_sim = self.quant_input(x)

                # calculate similarity and store them # 计算量化后的相似度
                out_sim = F.conv2d(x_sim, w_sim, bias_sim, self.stride, self.padding, self.dilation, self.groups)  # shape: B,parallel_eq_n*oc,fw,fh
                out_sim = torch.cat(torch.chunk(out_sim.unsqueeze(1), chunks=p_ed - p_st, dim=2), dim=1)  # shape: B,parallel_eq_n,oc,fw,fh
                similarity = self._get_similarity(raw_out, out_sim)  # shape: B,parallel_eq_n,oc,fw,fh # 计算相似度

                similarity = torch.mean(similarity, [3, 4])  # shape: B,parallel_eq_n,oc
                similarity = torch.sum(similarity, dim=0, keepdim=True)  # shape: (1,parallel_eq_n) or (1,parallel_eq_n,oc)

                similarities.append(similarity)
            similarities = torch.cat(similarities, dim=1)  # shape: (1,eq_n) or (1,eq_n,oc)
            batch_similarities.append(similarities)
        batch_similarities = torch.cat(batch_similarities, dim=0).sum(dim=0, keepdim=False)  # shape: (eq_n) or (eq_n,oc)
        _, best_index = torch.topk(batch_similarities, k=topk, dim=0)  # 取最优的top k候选
        best_index = best_index.view(topk, -1, 1)  # 重塑张量
        if topk == 1:
            tmp_w_scale = torch.gather(weight_scale_candidates, dim=0, index=best_index)
            tmp_w_zero_point = torch.gather(weight_zero_point_candidates, dim=0, index=best_index)
            self.w_quantizer.scale.data.copy_(tmp_w_scale.squeeze(dim=0))
            self.w_quantizer.zero_point.data.copy_(tmp_w_zero_point.squeeze(dim=0))
        return best_index

    def weight_fpcs(self, fpcs_width=16, steps=4, search_strategy=None):
        fpcs_new_cnt = int(self.eq_n / fpcs_width)  # 每轮需要生成的候选数量
        weight_scale_candidates, weight_zero_point_candidates = self.calculate_percentile_weight_candidates()  # 通过百分位数计算出权重scale和zp的候选值
        delta_scale = weight_scale_candidates[1:2] - weight_scale_candidates[0:1]
        topk_index = search_strategy(self, weight_scale_candidates, weight_zero_point_candidates, topk=fpcs_width)
        topk_scale_candidates = torch.gather(weight_scale_candidates, dim=0, index=topk_index)
        topk_zp_candidates = torch.gather(weight_zero_point_candidates, dim=0, index=topk_index)
        remain_steps = steps - 1  # 这里-1合适吗？
        while remain_steps > 0:
            delta_scale_candidates = (torch.linspace(0, 1, steps=fpcs_new_cnt).cuda()[:, None, None] - 0.5) * delta_scale
            delta_scale = delta_scale / (fpcs_new_cnt - 0.5)  # 缩小搜索范围：一开始的搜索范围要大，然后就可以减小搜索范围了，搜索的次数应该是不变的
            weight_scale_candidates = (topk_scale_candidates.unsqueeze(1) + delta_scale_candidates.unsqueeze(0)).reshape(-1, *weight_scale_candidates.shape[1:])
            weight_zero_point_candidates = topk_zp_candidates.repeat_interleave(fpcs_new_cnt, dim=0)
            topk_index = search_strategy(self, weight_scale_candidates, weight_zero_point_candidates, topk=1 if remain_steps == 1 else fpcs_width)
            if remain_steps > 1:
                topk_scale_candidates = torch.gather(weight_scale_candidates, dim=0, index=topk_index)
                topk_zp_candidates = torch.gather(weight_zero_point_candidates, dim=0, index=topk_index)
            remain_steps -= 1

    # def calculate_w_params_adalog(self):
    #     '''这是adlog中对权重使用fpcs搜索量化参数'''
    #     self._initialize_calib_parameters()  # 初始化校准参数？
    #     weight_scale_candidates, weight_zero_point_candidates = self.calculate_percentile_weight_candidates()
    #     '''量化参数候选值为什么是3维的张量？——实际上是eq_n组通道级量化参数'''
    #
    #     '''这里选取倒数第二组候选值，是一组偏保守的候选值'''
    #     self.w_quantizer.scale.data.copy_(weight_scale_candidates[-2])
    #     self.w_quantizer.zero_point.data.copy_(weight_zero_point_candidates[-2])
    #     self.w_quantizer.inited = True
    #
    #     for e in range(self.search_round):  # 多轮搜索，先处理权重，再处理激活值
    #         if self.fpcs:  # 启用FPCS
    #             self.weight_fpcs(steps=self.steps, search_strategy=GPTQConv2D._search_best_w_scale)
    #         else:  # 普通搜索
    #             self._search_best_w_scale(weight_scale_candidates, weight_zero_point_candidates)
    #         if self.a_quantizer.n_bits < 8:  # 这里是不成立的
    #             # self._search_best_a_scale(input_scale_candidates)
    #             pass
    #         else:  # 那就说明只对权重量化参数进行一次搜索？
    #             break
    # def params_searching_gptq(self):
    #     '''原来未分离的参数搜索，在计算完Hessian矩阵后进行量化，量化的过程中进行参数搜索，这样不太好。'''
    #     self.calculate_w_params() # 使用AdaLog中的方法计算量化参数。
    #     self.calculate_Hessian() # 计算Hessian矩阵
    #     '''这里传入的权重数据一定是卷积层的权重，所以，只要把quant()方法中的判断语句拿到外面来即可。'''
    #     weight = self.weight.data.clone()
    #
    #     # print(f"self.raw_input.device = {self.raw_input.device.type}")
    #     W = weight.flatten(1) # [oc,ic*kh*kw]
    #
    #     W_sim = self.w_quantizer.quant(self.H, W,self.columns,weight_flag=True) # 直接量化权重，这个方法里面包含了初始化量化参数的部分。而且默认就是通道量化
    #     # print(f"W.shape = {W.shape}")
    #     # print(f"W_sim.shape = {W_sim.shape}")
    #     weight_sim = W_sim.reshape(weight.shape) # oc,ic,kh,kw
    #     # self.weight.data = weight_sim
    #     self.quantized_weight = weight_sim # 这里保存的是量化后的权重
    #     self.calibrated = True
    #     '''下面的几行代码验证第一个卷积层的输入数据是不是和原始的图像数据一致？——raw_input与校准过程中的输入数据完全一样。'''
    #     # print(f"第一个卷积层的raw_input.shape: {self.raw_input.shape}")

    """2025年8月22日：下面的方法是将参数搜索和量化部分解耦"""
    def calculate_w_params(self,weight):
        ''''''
        x = weight.data.clone()
        dev = x.device
        # self.maxq = self.maxq.to(dev) # 确保同设备
        self.maxq = torch.tensor(2 ** self.w_bit - 1).to(dev)
        shape = x.shape  # 一开始未进行处理的原始权重。
        '''要按照量化的方式，分别对权重和激活值的维度进行处理'''
        if self.channel_wise:  # 逐通道处理
            x = x.flatten(1) # 逐通道为什么要直接展平？
        else:  # 非通道，即全局量化；无论是权重还是激活，全局量化都是直接展平。
            x = x.flatten().unsqueeze(0)  # 全局量化也是同样的操作，先全局展平，然后再添加一个维度
        tmp = torch.zeros(x.shape[0], device=dev)  # 创建一个与x第0维大小相同的全零张量，对应的是权重的输出维度（oc或者out_feature）
        # 在x的第二维取最小值，然后拿到第一个元素，与0相比？
        xmin = torch.minimum(x.min(1)[0], tmp)  # x的每一行数据（第1维）中取一个最小值，且最小值要保证小于0；这个对应的是每个通道的最小值
        xmax = torch.maximum(x.max(1)[0], tmp)  # x的每一行数据中取一个最大值，且最大值保证要大于0；同上，最大值
        '''取最值的解释：二维的矩阵，第0维是输出维度，第1维是输入（Linear）或者特征维度（Conv2D），最值是每个输出维度的最值。'''
        if self.symmetric:  # 对称量化
            xmax = torch.maximum(torch.abs(xmin), xmax)  # 最大值设置为xmin和xmax绝对数值的最大值，而且是针对每一行（第0维）获取最大值；这里的xmax被重新赋值了
            tmp = xmin < 0  # 标记哪些哪些行的值出现了负数，需要进行调整；因为如果出现小于0的情况，就不满足对称的条件——为什么出现负值就不满足对称的条件？-128~128不也是对称的吗？
            if torch.any(tmp):  # 检查tmp中有无满足xmin < 0的标记
                xmin[tmp] = -xmax[tmp]  # 将所有负值区域的最小值设为-xmax，最终得到对称范围：[-xmax, +xmax]
        tmp = (xmin == 0) & (xmax == 0)  # 处理全零数据，这个对应的范围是[0,0]，但是这种情况会出现除0，所以还要处理。
        # 将全零区域的人工设置为[-1, +1]的虚拟范围，避免计算缩放因子时出现除0错误。
        xmin[tmp] = -1
        xmax[tmp] = +1
        '''下面的代码是对上面注释掉的代码进行的精简，删除三值量化的判断。'''
        self.w_quantizer.scale = nn.Parameter((xmax - xmin) / (2 ** self.w_bit - 1))  # scale的计算对于对称还是非对称可以通用
        if self.symmetric:
            self.w_quantizer.zero_point = nn.Parameter(torch.zeros_like(self.w_quantizer.scale))  # 对称量化还需要零点吗？
        else:  # 非对称量化，计算每个通道/维度的自适应零点，使量化能够覆盖原始数据的最小值。
            self.w_quantizer.zero_point = nn.Parameter(torch.round(- xmin / self.w_quantizer.scale))  # 这里如果使用round，那么在后向传播的时候会导致无法计算梯度。但是，我需要后向传播梯度吗？
        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            # 网格搜索，最大搜索比×网格点，实际上搜不到全部的网格
            for i in range(int(self.maxshrink * self.grid)):  # 默认是100个点，收缩比为0.8，乘积为80
                p = 1 - i / self.grid  # 先算除法，再算减法。收缩比衰减，从1衰减到0.21，每次衰减0.01
                '''逐步缩小量化区间 [xmin, xmax] 的范围'''
                xmin1 = p * xmin
                xmax1 = p * xmax
                # 重新计算量化参数
                # scale1 = (xmax1 - xmin1) / self.maxq
                scale1 = (xmax1 - xmin1) / (2 ** self.w_bit - 1)
                zero1 = torch.round(-xmin1 / scale1) if not self.symmetric else self.w_quantizer.zero_point
                q = torch.clamp(torch.round(x / scale1.unsqueeze(1)) + zero1.unsqueeze(1), 0, self.maxq)  # 这里是非对称量化吧？
                q_deq = scale1.unsqueeze(1) * (q - zero1.unsqueeze(1))  # 反量化？
                q_res = q_deq - x  # 与全精度的值做差
                q_res.abs_()  # 取绝对值
                q_res.pow_(self.norm)  # self.norm=2.4，计算差值的范数
                err = torch.sum(q_res, 1)  # 累积误差
                tmp = err < best  # 当前的量化误差比记录的最优值小，就更新最优值和量化参数
                '''逐行找到合适的量化参数'''
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.w_quantizer.scale[tmp] = scale1[tmp]
                    self.w_quantizer.zero_point[tmp] = zero1[tmp]
        if not self.channel_wise:
            '''# 非通道量化，也就是全局量化。但是，最后的两条语句还是把量化参数repeat成了通道级'''
            tmp = shape[0]  # 权重原始的shape，第0维表示oc
            self.w_quantizer.scale = self.w_quantizer.scale.repeat(tmp)
            self.w_quantizer.zero_point = self.w_quantizer.zero_point.repeat(tmp)  # 这里只是对齐第一维，本质上是将一个标量扩展成一个一维向量，而下面的代码是将1维广播成对应的维度
        '''下面的几个if是用来恢复原来的张量形状，因为在一开始的时候对张量的形状进行了reshape'''

        self.w_quantizer.scale = nn.Parameter(self.w_quantizer.scale.reshape(-1,1))
        self.w_quantizer.zero_point = nn.Parameter(self.w_quantizer.zero_point.reshape(-1,1))
        # 这里就是4维的量化参数了。
        return  # 这里怎么有个return？——明白了，下面的都是输入值的维度恢复，如果是权重，下面的代码就没用了

    def params_searching(self):
        '''重写这个参数搜索方法，将量化的代码分离出来。卷积层不需要区分n_V，是对整个权重矩阵进行的量化。'''
        weight = self.weight.data.clone()
        self.calculate_w_params(weight=weight) # 使用AdaLog中的方法计算量化参数。
        self.calculate_Hessian() # 计算Hessian矩阵
        '''这里传入的权重数据一定是卷积层的权重，所以，只要把quant()方法中的判断语句拿到外面来即可。'''
        self.calibrated = True
        '''下面的几行代码验证第一个卷积层的输入数据是不是和原始的图像数据一致？——raw_input与校准过程中的输入数据完全一样。'''
        # print(f"第一个卷积层的raw_input.shape: {self.raw_input.shape}")
