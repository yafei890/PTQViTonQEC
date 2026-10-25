import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
import sys
import logging
from mpmath import zeros
# from xlwings.pro.reports.filters import scale

from quantizers.uniform import UniformQuantizer
# from quantizers.gptq_quant import GPTQQuantizer
from quantizers.gptq import GPTQQuantizer
from quantizers.logarithm import ShiftAdaLogQuantizer,ShiftLog2Quantizer,ShiftLogSqrt2Quantizer
import math
import time
import os
# from test_tensor_is_same import weight

'''
如何构建？
Yuan和Zhu构建量化层主要是为了搜索量化参数，然后进行量化。我这里不需要很复杂的搜参数过程，而且能将GPTQ中对每一层的量化代码拿过来。
'''


class MinMaxQuantLinear(nn.Linear):
    '''Linear、Conv和Matmul的量化层都是从Min-Max量化开始的，这是一个基础的量化方法，只需要知道张量的最大最小值就行了。'''

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 mode="raw",
                 w_bit=4,
                 a_bit=4):
        super().__init__(in_features, out_features, bias)
        self.mode = mode
        # self.w_quantizer = UniformQuantizer(n_bits=w_bit, symmetric=True, channel_wise=True)
        self.w_quantizer = GPTQQuantizer(n_bits=w_bit,symmetric=False,channel_wise=True)
        self.a_quantizer = UniformQuantizer(n_bits=a_bit, symmetric=False, channel_wise=True)
        self.raw_input = None
        self.raw_out = None
        self.tmp_input = None
        self.tmp_out = None
        self.calibrated = False
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.quantized_weight = torch.zeros(self.weight.shape)  # 创建一个新的张量，用于保存量化后的权重
        # self.quantized_weight = self.weight.clone().detach()
        # self.quantized_weight.to('cuda' if torch.cuda.is_available() else 'cpu')
        # self.register_buffer('quantized_weight', self.weight.detach().clone())
        # self.H = torch.zeros(self.columns, self.columns).to(self.device)
        # self.columns = self.weight.shape[1]  # 列数

    def forward(self, x):
        if self.mode == 'raw':
            # print("Linear层能够执行到未量化的前向传播")
            out = F.linear(x, self.weight, self.bias)
        elif self.mode == "quant_forward":
            out = self.quant_forward(x)
            # sys.exit(1137)
        # elif self.mode == 'debug_only_quant_weight':  # 仅量化权重调试
        #     out = self.debug_only_quant_weight(x)
        elif self.mode == 'debug_only_quant_act':  # 仅量化激活调试
            out = self.debug_only_quant_act(x)
        elif self.mode == 'quant_weight_only':
            out = self.quant_weight_only(x)
        else:
            raise NotImplementedError

        return out

    # def quant_weight_bias(self):
    #     '''这里只能把权重传进去，但是我还需要Hessian矩阵和列数'''
    #     print(f"会调用minmax方法的quant_weight_bias吗？") # 没有调用这个方法。
    #     w_sim = self.w_quantizer(self.weight)
    #     return w_sim, self.bias if self.bias is not None else None
    '''2025年8月27日：只量化权重，看一下相比之前版本的代码，到底是那里出了问题导致的准确率下降'''
    def quant_weight_only(self,x):
        # x = x - 0.16997124254703522
        w_sim, bias_sim = self.quant_weight_bias()
        out = F.linear(x, w_sim, bias_sim)
        return out

    def quant_input(self, x):
        return self.a_quantizer(x)

    def quant_forward(self, x):
        '''对权重、偏差和激活值都进行量化'''
        assert self.calibrated, f"Module should be calibrated before run quant_forward for {self}"
        # print(f"self.w_quantizer.hessian = {self.w_quantizer.hessian}")
        w_sim,bias_sim = self.quant_weight_bias()
        x_sim = self.quant_input(x)
        out = F.linear(x_sim, w_sim, bias_sim)
        return out
    def debug_only_quant_act(self, x):
        x_sim = self.quant_input(x)
        out = F.linear(x_sim, self.weight, self.bias)
        return out


class GPTQLinear(MinMaxQuantLinear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = False,
                 w_bit=4,
                 a_bit=4,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 n_V=1,
                 mode='raw',
                 groups: int = 1,
                 channel_wise: bool = False,
                 mse : bool = True,
                 symmetric : bool = False,
                 blocksize=128,  # 论文中的block
                 percdamp=.01,  # 添加到 Hessian 对角线的阻尼比例（防止奇异）
                 groupsize=-1,  # 对权重列进行分组以共享量化参数，-1 表示不分组
                 actorder=False,  # 是否按照激活重要性对列重新排序（论文提到的 activation-ordering）
                 static_groups=False,  # 是否对组的量化器静态缓存
                 ):
        super().__init__(in_features, out_features, bias ,device)
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.device = device
        self.mode = mode
        self.n_V = n_V
        self.raw_input = None
        self.raw_out = None
        self.tmp_input = None
        self.tmp_out = None
        self.calibrated = False
        self.blocksize = 128
        self.percdamp = 0.01
        self.groupsize = -1
        self.actorder = False
        self.static_groups = False
        # 下面的两个参数由量化器指定，不需要在量化层中定义。
        self.channel_wise = True
        self.symmetric = False
        self.mse = True


        '''下面的4行代码是在对attn.qkv和mlp.qkv使用AdaLog中的百分位数计算量化参数时对这个类新增的代码，后面可能需要重新封装一个类'''
        self.calib_batch_size = 32
        self.eq_n = 128



        # 实例化Linear的权重量化器，这些参数都是量化器本身的，跟Linear还是Conv无关。
        self.w_quantizer = GPTQQuantizer(n_bits=self.w_bit,channel_wise=self.channel_wise,mse=self.mse,symmetric=self.symmetric)
        # self.w_quantizer
        self.nsamples = 0

        '''创建Hessian矩阵。这里应该是要改的，因为Linear和Conv的权重不一样'''
        W = self.weight.data.clone()
        # W = W.flatten(1) # Linear不需要展平
        self.rows = W.shape[0]  # 行数
        self.columns = W.shape[1]  # 列数
        self.H = torch.zeros(self.columns, self.columns).to(self.device)
        '''2025年8月21日：分离量化参数搜索与量化过程需要修改的参数'''
        self.norm = 2.4
        self.maxshrink = .8
        self.grid = 100

    """下面的方法是2025年8月21日重写forward方法时使用的代码："""
    # def quant_weight_bias(self):
    #     """这个方法是给前向传播用的，在这个方法中要收集舍入值。——不对，舍入值不应该在这里收集。"""
    #     W = self.weight.data.clone()
    #     quantized_weight = []
    #     # rest = []  # 2025年8月29日新增，统计量化时舍入的余数。——2025年9月1日：不能在这个方法里面统计。
    #     if self.n_V == 3:
    #         scale = self.w_quantizer.scale.data.clone()
    #         zero_point = self.w_quantizer.zero_point.data.clone()
    #         with (torch.no_grad()):
    #             qkv = torch.chunk(W, self.n_V, dim=0)  # 3 x [192, 192]
    #             for i,weight in enumerate(qkv):
    #                 print(f"循环qkv子矩阵....")
    #                 self.w_quantizer.scale = nn.Parameter(scale[i])
    #                 self.w_quantizer.zero_point = nn.Parameter(zero_point[i])
    #                 weight_sim = self.w_quantizer(weight, self.H, self.columns)
    #                 quantized_weight.append(weight_sim) # 子矩阵量化，然后在拼接回去
    #                 # rest.append(self.w_quantizer.rest.data)
    #             '''堆叠成三维张量'''
    #         self.w_quantizer.scale = nn.Parameter(scale)
    #         self.w_quantizer.zero_point = nn.Parameter(zero_point)
    #         weight_sim = torch.cat(quantized_weight, dim=0)  # [576, 192]
    #         # self.w_quantizer.rest = torch.cat(rest,dim=0) # 将权重的舍入余数拼接回去
    #     else:
    #         with (torch.no_grad()):
    #             weight_sim = self.w_quantizer(self.weight, self.H, self.columns)
    #             # rest.append(self.w_quantizer.rest.data)
    #             # self.w_quantizer.rest = torch.cat(rest,dim=0) # 将权重的舍入余数拼接回去
    #     return weight_sim, self.bias if self.bias is not None else None

    def cosine_similarity(self, tensor1, tensor2, eps=1e-8):
        tensor1_flat = tensor1.flatten()
        tensor2_flat = tensor2.flatten()
        return F.cosine_similarity(tensor1_flat.unsqueeze(0), tensor2_flat.unsqueeze(0), dim=1, eps=eps).item()

    def quant_weight_bias(self):
        """这个方法是给激活值参数搜索用的，与上面的方法量化方式相同，只是不需要收集舍入值。"""
        W = self.weight.clone()
        W_origin = self.weight.clone()
        quantized_weight = []
        if self.n_V == 3:
            scale = self.w_quantizer.scale.data.clone()
            zero_point = self.w_quantizer.zero_point.data.clone()
            qkv = torch.chunk(W, self.n_V, dim=0)  # attn.qkv: [3,192,192]
            for i,weight in enumerate(qkv):
                self.w_quantizer.set_index(i)
                self.w_quantizer.scale = nn.Parameter(scale[i])
                self.w_quantizer.zero_point = nn.Parameter(zero_point[i])
                W_sim = self.w_quantizer(weight)
                quantized_weight.append(W_sim) # 子矩阵量化，然后在拼接回去
            '''堆叠成三维张量'''
            self.w_quantizer.scale = nn.Parameter(scale)
            self.w_quantizer.zero_point = nn.Parameter(zero_point)
            W_sim = torch.cat(quantized_weight, dim=0)  # [576, 192]
        else:
            W_sim = self.w_quantizer(W)
        # print(f"W.shape = {W.shape}")
        return W_sim, self.bias if self.bias is not None else None


    def calculate_w_params(self,weight):
        '''2025年8月21日09点10分：目前这个方法只能针对一个权重子矩阵进行量化，无法量化整个权重矩阵。——反正这个方法写的不对'''
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

        shape = [-1] + [1] * (len(shape) - 1)  # 结果是[-1,1,1,1]吗？——应该不是，原始权重是2维的，这里应该也是二维的。

        self.w_quantizer.scale = nn.Parameter(self.w_quantizer.scale.reshape(shape))
        self.w_quantizer.zero_point = nn.Parameter(self.w_quantizer.zero_point.reshape(shape))
        return  # 这里怎么有个return？——明白了，下面的都是输入值的维度恢复，如果是权重，下面的代码就没用了


    def calculate_Hessian(self):
        '''这是Linear层中的计算Hessian矩阵的方法，可以进行一下简化。self是GPTQLinear'''
        # print(f"type of self: {type(self).__name__}")
        inp = self.raw_input.to(self.device).clone() # 32,197,192
        weight = self.weight.data.clone() # 576,192
        self.w_quantizer.rows = weight.shape[0]
        self.w_quantizer.cols = weight.shape[1]
        self.w_quantizer.hessian = torch.zeros(self.w_quantizer.cols, self.w_quantizer.cols).to(self.device) # 192,192
        # print(f"calculate_Hessian方法中的inp.shape：{inp.shape}") # 输入不会为0.
        tmp = inp.shape[0]  # 拿到批次数据
        # 处理不同类型的输入，目的是转换成二维的矩阵
        if isinstance(self, nn.Linear):
            '''在NLP中，Linear层处理之后，这两种层的维度都是[batch, seq_len, dim(或者叫hidden_size)]'''
            # print(f"inp.shape: {inp.shape}")
            # sys.exit()
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))  # shape:[D,B*T]，这时候每一行是一个样本
            if len(inp.shape) == 4:
                inp = inp.reshape(-1, inp.shape[-1])  # [B*H*W, C] = [25088, 384]
            # inp = inp.t() # 在swin transformer中，直接转置会报错，因为downsample.reduction层的输入是4维。
            try:
                inp = inp.t()  # 转置一下，shape:[B*T,D]，每一列是一个样本
            except:
                print(f"type(self) is {type(self).__name__}")
                print(f"inp.shape: {inp.shape}")
                print(f"self.raw_input.shape: {self.raw_input.shape}")
                print(f"self.raw_out.shape: {self.raw_out.shape}")
                sys.exit()
        # 更新Hessian矩阵，论文中的更新原理是高斯消元，但是这里好像没体现出来？
        self.w_quantizer.hessian *= self.nsamples / (self.nsamples + tmp)  # 滑动平均 # 这里的self.H就是0
        self.nsamples += tmp  # 更新总样本数 # 这里就非0了
        inp = math.sqrt(2 / self.nsamples) * inp.float()  # 归一化缩放
        self.w_quantizer.hessian += inp.matmul(inp.t())  # 对应论文中H=xx^T。 这里乘法用的就是数学意义上的矩阵乘，但依然能实现H的计算。
        # print(f"计算Hessian矩阵时是否为0：{torch.equal(self.H,torch.zeros_like(self.H))}")
        # print(f"计算Hessian矩阵计算时的结果：{self.H}")


    """2025年8月14日10点25分：下面的代码是重写Linaer权重量化参数的搜索"""
    def _get_similarity(self, tensor_raw, tensor_sim):
        return -(tensor_raw - tensor_sim) ** 2  # 计算原始张量和量化后张量的负均方误差。

    def _initialize_calib_parameters(self):
        '''计算可用的显存大小，从而确定并行的参数量。根据GPU显存动态计算并行处理的候选参数数量。这个不是很关键，如果本地16GB的GPU内存不够用的话，就用服务器。'''
        self.calib_size = self.raw_input.shape[0]
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            memory = props.total_memory // 2
        else:
            raise EnvironmentError("CUDA is not available on this system")
        numel = (8 * self.raw_input[:self.calib_batch_size].numel() +
                 16 * self.raw_out[:self.calib_batch_size].numel()) # number of parameters on GPU
        self.parallel_eq_n = int((memory / 4) // numel)
        self.parallel_eq_n = math.ceil(self.eq_n * 1.0 / math.ceil(self.eq_n * 1.0 / self.parallel_eq_n))


i=0
class GPTQAsymmetricallyBatchingQuantLinear(GPTQLinear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = False,
                 w_bit=4,
                 a_bit=4,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 n_V=1,
                 mode='raw',
                 calib_batch_size=32,
                 eq_n = 128,
                 search_round = 3,
                 fpcs = False,
                 # groups: int = 1,
                 # channel_wise: bool = False,
                 # mse: bool = True,
                 # symmetric: bool = False,
                 # blocksize=128,  # 论文中的block
                 # percdamp=.01,  # 添加到 Hessian 对角线的阻尼比例（防止奇异）
                 # groupsize=-1,  # 对权重列进行分组以共享量化参数，-1 表示不分组
                 # actorder=False,  # 是否按照激活重要性对列重新排序（论文提到的 activation-ordering）
                 # static_groups=False,  # 是否对组的量化器静态缓存
                 ):
        super().__init__(in_features, out_features, bias=bias, mode=mode, w_bit=w_bit, a_bit=a_bit,n_V=n_V)
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.device = device
        self.raw_input = None
        self.raw_out = None
        self.tmp_input = None
        self.tmp_out = None
        self.mode = mode
        self.calibrated = False
        # self.blocksize = blocksize
        # self.percdamp = percdamp
        # self.groupsize = groupsize
        # self.actorder = actorder
        # self.static_groups = static_groups
        # self.channel_wise = channel_wise
        # self.mse = mse
        # self.symmetric = symmetric
        '''下面的4行代码是在对attn.qkv和mlp.qkv使用AdaLog中的百分位数计算量化参数时对这个类新增的代码，后面可能需要重新封装一个类'''
        self.calib_batch_size = calib_batch_size
        self.eq_n = eq_n
        self.search_round = search_round
        # self.crb_rows = out_features // n_V
        '''下面的代码是在FPCS搜索时使用的'''
        self.fpcs = fpcs
        self.steps = 6

        # 实例化Linear的权重量化器，这些参数都是量化器本身的，跟Linear还是Conv无关。
        '''删除原来权重和激活值的均匀量化器，重新构建非对称非通道量化器'''
        del self.a_quantizer
        self.a_quantizer = UniformQuantizer(n_bits=a_bit, symmetric=False, channel_wise=False)
        '''这里是量化参数的维度，对激活值采用的全局量化，维度是1；权重的维度是(n_V,crb_rows,1)'''
        '''原版
        self.a_quantizer.scale = nn.Parameter(torch.zeros((1)))
        self.a_quantizer.zero_point = nn.Parameter(torch.zeros((1)))
        # 如果是使用gptq的量化参数搜索方法，那么就不能使用下面的参数。
        self.w_quantizer.scale = nn.Parameter(torch.zeros((n_V, self.out_features //self.n_V, 1)))
        self.w_quantizer.zero_point = nn.Parameter(torch.zeros((n_V, self.out_features //self.n_V, 1)))
        '''

        # 使用37作为特殊标记值
        self.a_quantizer.scale = nn.Parameter(torch.full((1,), 37.0))
        self.a_quantizer.zero_point = nn.Parameter(torch.full((1,), 37.0))
        '''如果是使用gptq的量化参数搜索方法，那么就不能使用下面的参数。'''
        self.w_quantizer.scale = nn.Parameter(torch.full((n_V, self.out_features // self.n_V, 1), 37.0))
        self.w_quantizer.zero_point = nn.Parameter(torch.full((n_V, self.out_features // self.n_V, 1), 37.0))
        # self.w_quantizer.rest = None # 还是用来占位
        # if self.n_V == 3:
        #     self.w_quantizer.scale = nn.Parameter(torch.full((n_V, self.out_features // self.n_V, 1), 37.0))
        #     self.w_quantizer.zero_point = nn.Parameter(torch.full((n_V, self.out_features // self.n_V, 1), 37.0))
        # else:
        #     self.w_quantizer.scale = nn.Parameter(torch.full((self.out_features, 1), 37.0))
        #     self.w_quantizer.zero_point = nn.Parameter(torch.full((self.out_features, 1), 37.0))
        """2025年8月25日：在此类中重新初始化一个Hessian矩阵""" # 好像不是这里的问题
        W = self.weight.data.clone()
        # W = W.flatten(1) # Linear不需要展平
        self.rows = W.shape[0]  # 行数
        self.columns = W.shape[1]  # 列数
        self.w_quantizer.hessian = torch.zeros(self.columns, self.columns).to(self.device)

    def _initialize_activation_scale(self):
        tmp_a_scales = []
        tmp_a_max, tmp_a_min = [], []
        for b_st in range(0, self.raw_input.shape[0], self.calib_batch_size):
            b_ed = min(self.raw_input.shape[0], b_st + self.calib_batch_size)
            x_ = self.raw_input[b_st:b_ed].cuda()
            '''激活值初始化要分是否通道级量化，是：逐通道构建缩放因子和零点；否：全局共用一个缩放因子和零点'''
            if self.a_quantizer.channel_wise:
                '''在所有非最后一个维度上求最大值/最小值。全连接层常见的维度是[batch_size, features]，最后一个维度是in_features，
                通道级的量化就是按照输入通道进行划分，在除通道维度外的所有其他维度上求极值'''
                a_max = x_.abs().amax([i for i in range(x_.ndim - 1)], keepdim=False).detach().view(1, -1)
                a_min = x_.abs().amin([i for i in range(x_.ndim - 1)], keepdim=False).detach().view(1, -1)
            else:  # 全局量化
                a_max = x_.abs().max().detach().view(1, 1)  # 第一维是为了后面合并批次数据
                a_min = x_.abs().min().detach().view(1, 1)
            '''这里存储的是当前batch的统计量'''
            tmp_a_max.append(a_max)
            tmp_a_min.append(a_min)
        tmp_a_max = torch.cat(tmp_a_max, dim=0).amax(dim=0, keepdim=False)  # 所有batch中取最大值
        tmp_a_min = torch.cat(tmp_a_min, dim=0).amin(dim=0, keepdim=False)  # 所有batch中取最小值
        self.a_quantizer.scale.data.copy_((tmp_a_max - tmp_a_min) / (2 * self.a_quantizer.n_levels - 1))
        self.a_quantizer.zero_point.data.copy_(-tmp_a_min / self.a_quantizer.scale)
        self.a_quantizer.inited = True

    def _search_best_a_scale_self(self, input_scale_candidates, input_zero_point_candidates, topk=1):
        '''权重的初始化没有用到校准数据，激活值的初始化好像用到了？'''
        batch_similarities = []  # similarities, need to concatenate and calculate sum (equivalent to mean with argmax)
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            # print(f"type of self.calib_batch_size: {type(self.calib_batch_size).__name__}")
            # print(f"type of self.calib_size: {type(self.calib_size).__name__}")
            # sys.exit()
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            x = self.raw_input[b_st:b_ed].cuda()
            raw_x = self.raw_input[b_st:b_ed].cuda().unsqueeze(-1)  # shape: b,*,in_features,1
            similarities = []
            for p_st in range(0, self.eq_n, self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st + self.parallel_eq_n)
                cur_a_scale = input_scale_candidates[:, p_st:p_ed]
                cur_a_zero_point = input_zero_point_candidates[:, p_st:p_ed]
                # quantize input
                x_sim = x.unsqueeze(-1)  # shape: B,*,in_features,1
                try:
                    x_quant = ((x_sim / cur_a_scale).round_() + cur_a_zero_point).clamp_(0,2 * self.a_quantizer.n_levels - 1)  # shape: B,*,in_features,parallel_eq_n
                except:
                    print(f"这里报错：")
                    print(f"type of self : {type(self).__name__}")
                    '''AdaLog中的GELU量化更本就不会调用到此类中的这个方法。'''
                x_dequant = (x_quant - cur_a_zero_point) * cur_a_scale  # shape: B,*,in_features,parallel_eq_n
                similarity = self._get_similarity(raw_x, x_dequant)  # shape: b,*,in_features,parallel_eq_n
                '''合并中间的维度？'''
                if len(similarity.shape) > 3:  # 保留第一维和最后两维
                    similarity = torch.mean(similarity, dim=list(
                        range(1, len(similarity.shape) - 2)))  # shape: b, in_features, parallel_eq_n
                if not self.a_quantizer.channel_wise:  # 非通道级量化，把通道维度合并掉，但是保留维度
                    similarity = torch.mean(similarity, dim=1, keepdim=True)  # shape: b, 1, parallel_eq_n
                similarity = torch.sum(similarity, dim=0,
                                       keepdim=True)  # 3维张量，但是有两种形式：[1,in_features,parallel_eq_n]或[1,1,parallel_eq_n]
                similarities.append(similarity)
            # store best input scale and store in tmp_a_scale
            similarities = torch.cat(similarities, dim=-1)  # 合并掉并行计算：shape: 1, in_features, eq_n
            batch_similarities.append(similarities)
        batch_similarities = torch.cat(batch_similarities, dim=0).sum(dim=0, keepdim=False)  # shape: in_features, eq_n
        _, best_index = torch.topk(batch_similarities, k=topk, dim=-1)  # shape: in_features, topk；这里的in_features有可能是1
        if topk == 1:
            tmp_a_scale = torch.gather(input_scale_candidates, dim=-1, index=best_index)
            tmp_a_zero_point = torch.gather(input_zero_point_candidates, dim=-1, index=best_index)
            '''从这里给量化器中的量化参数进行了赋值，量化时可直接调用。'''
            self.a_quantizer.scale.data.copy_(tmp_a_scale.squeeze(-1))
            self.a_quantizer.zero_point.data.copy_(tmp_a_zero_point.squeeze(-1))
            self.a_quantizer.inited = True
            self.a_quantizer.to(self.device)
        return best_index

    def calculate_percentile_activation_candidates(self, l=0.9, r=1.0):
        num_zp = min(16, self.a_quantizer.n_levels * 2)  # 精度只要大于等于3bit，就只保留16个zp候选值点
        num_scale = int(self.eq_n / num_zp)  # 保证zp和scale候选值点的乘积是恒定的
        # percentiles_uppers, percentiles_lowers = [], [] # 这行代码应该是从别的地方复制来的，在此方法中没用到
        pct = torch.tensor([l, r])
        x = self.raw_input.cuda()  # 这里对x的维度只要求最后一维是in_features
        '''以下两个参数用于初始化标志位和最小批量大小，用于处理大张量'''
        tensor_too_large = True
        mini_batch_size = 1

        if self.a_quantizer.channel_wise:  # 通道级量化，将输入张量进行展平，只关注最后一个in_features维度，计算完再交换两个维度
            a_uppers_candidates = torch.quantile(x.view(-1, x.shape[-1]), pct.to(x.device), dim=0).transpose(0,1)  # shape: in_features, 2
            a_lowers_candidates = torch.quantile(x.view(-1, x.shape[-1]), (1 - pct).to(x.device), dim=0).transpose(0,1)  # shape: in_features, 2
        else:  # 非通道级量化，就得考虑张量是否过大，能不能处理。先尝试计算全局的百分位数，计算失败则增加批量的大小
            while tensor_too_large:
                try:
                    '''按照批次分别计算均值作为百分位数的计算结果'''
                    a_uppers_candidates = torch.quantile(x.view(mini_batch_size, -1), pct.to(x.device), dim=-1).mean(dim=-1).unsqueeze(0)  # shape: 1, 2
                    a_lowers_candidates = torch.quantile(x.view(mini_batch_size, -1), (1 - pct).to(x.device),dim=-1).mean(dim=-1).unsqueeze(0)  # shape: 1, 2
                    tensor_too_large = False  # 能走到这一条语句就说明批次大小是合适的。
                except:
                    mini_batch_size *= 2
        '''按照批次分别计算保守区间范围和全部区间范围
        upper:[0.9 , 1.0], lower:[0.1 , 0]'''
        delta_min = a_uppers_candidates[:, 0:1] - a_lowers_candidates[:, 0:1]  # 范围：90%-10%
        delta_max = a_uppers_candidates[:, 1:] - a_lowers_candidates[:, 1:]  # 范围：100%-0%
        '''通过插值从保守范围向全部范围进行过渡'''
        splits = torch.linspace(0, 1, steps=num_scale).cuda()[None, :] * (delta_max - delta_min)
        a_scale_candidates = ((delta_min + splits).repeat(1, num_zp) / (2 * self.a_quantizer.n_levels - 1)).clamp(min=1e-4)
        '''激活值零点候选值的生成逻辑和前面的权重零点候选值的生成逻辑差不多'''
        zp_min = int(self.a_quantizer.n_levels - num_zp / 2)
        zp_max = int(self.a_quantizer.n_levels + num_zp / 2)
        zp_candidates = torch.tensor(range(zp_min, zp_max)).cuda()
        a_zero_point_candidates = zp_candidates.repeat_interleave(num_scale)[None, :]
        a_zero_point_candidates = a_zero_point_candidates.repeat(a_scale_candidates.shape[0], 1)
        return a_scale_candidates, a_zero_point_candidates

    def _search_best_a_scale(self, input_scale_candidates, input_zero_point_candidates, topk=1):
        batch_similarities = [] # similarities, need to concatenate and calculate sum (equivalent to mean with argmax)
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            x = self.raw_input[b_st:b_ed].cuda()
            raw_out_expanded = self.raw_out[b_st:b_ed].cuda().unsqueeze(-2) # shape: b,*,1,oc
            similarities = []
            for p_st in range(0,self.eq_n,self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st+self.parallel_eq_n)
                cur_a_scale = input_scale_candidates[:, p_st:p_ed]
                cur_a_zero_point = input_zero_point_candidates[:, p_st:p_ed]
                # quantize weight and bias
                '''这里需要重新调用权重的量化器。是重新调用还是直接读数据？'''
                w_sim, bias_sim = self.quant_weight_bias() # 也是均匀量化。为什么不使用上一步量化后的值？量化权重的时候需要量化完的激活值，而量化激活值的时候又需要量化后的权重？？？
                # w_sim, bias_sim = self.quant_weight_bias_for_a_params_calc()
                # w_sim = self.weight.data.clone()
                # bias_sim = self.bias.data.clone() if self.bias is not None else None

                # w_sim = self.w_quantizer.quant()
                # quantize input
                x_sim = x.unsqueeze(-1) # shape: B,*,in_features,1
                x_quant = ((x_sim / cur_a_scale).round_() + cur_a_zero_point).clamp_(0, 2 * self.a_quantizer.n_levels - 1) # shape: B,*,in_features,parallel_eq_n
                x_dequant = (x_quant - cur_a_zero_point) * cur_a_scale # shape: B,*,in_features,parallel_eq_n
                x_sim = x_dequant.permute(*list(range(len(x_sim.shape)-2)),-1,-2) # shape: B,*,parallel_eq_n,in_features；交换最后两个维度
                # calculate similarity and store them
                out_sim = F.linear(x_sim, w_sim, bias_sim)  # shape: b,*,parallel_eq_n,out_features
                similarity = self._get_similarity(raw_out_expanded, out_sim) # shape: b,*,parallel_eq_n,out_features
                similarity = torch.mean(similarity, dim=-1) # shape: B,*,parallel_eq_n
                if len(similarity.shape) > 2: # 只保留第一维和最后一维
                    similarity = torch.mean(similarity, dim=list(range(1,len(similarity.shape)-1))) # shape: b, parallel_eq_n
                similarity = torch.sum(similarity, dim=0, keepdim=True) # shape: 1, parallel_eq_n
                similarities.append(similarity)
            # store best input scale and store in tmp_a_scale
            similarities = torch.cat(similarities, dim=1) # shape: 1, eq_n
            batch_similarities.append(similarities)
        batch_similarities = torch.cat(batch_similarities, dim=0).sum(dim=0, keepdim=True) # shape: 1, eq_n
        _, best_index = torch.topk(batch_similarities, k=topk, dim=-1) # shape: 1, topk
        if topk == 1:
            tmp_a_scale = torch.gather(input_scale_candidates, dim=-1, index=best_index)
            tmp_a_zero_point = torch.gather(input_zero_point_candidates, dim=-1, index=best_index)
            self.a_quantizer.scale.data.copy_(tmp_a_scale.squeeze(-1))
            self.a_quantizer.zero_point.copy_(tmp_a_zero_point.squeeze(-1))
        return best_index

    def activation_fpcs(self, fpcs_width=16, steps=6, search_strategy=None):
        '''参考上一个对权重执行FPCS的方法，这个方法与上一个方法类似，区别应该只是张量维度的问题'''
        fpcs_new_cnt = int(self.eq_n / fpcs_width)
        a_scale_candidates, a_zero_point_candidates = self.calculate_percentile_activation_candidates()  # 通过百分位数进行初始化
        delta_scale = a_scale_candidates[:, 1:2] - a_scale_candidates[:, 0:1]
        '''下面的四行语句看着好像是从循环中单独拿出来的，反正是对第一次搜索特殊处理'''
        topk_index = search_strategy(self, a_scale_candidates, a_zero_point_candidates, topk=fpcs_width)  # 激活值的index应该是一个一维向量
        '''根据index拿到scale和zp的值，作为循环中的初始点，循环中在这些初始点附近进行精细化搜索'''
        topk_scale_candidates = torch.gather(a_scale_candidates, dim=-1, index=topk_index)
        topk_zp_candidates = torch.gather(a_zero_point_candidates, dim=-1, index=topk_index)
        remain_steps = steps - 1  # 前面的几行语句就相当于第一步搜索了
        while remain_steps > 0:
            delta_scale_candidates = (torch.linspace(0, 1, steps=fpcs_new_cnt).cuda()[None,:] - 0.5) * delta_scale  # shape:[in_features, fpcs_new_cnt]?
            delta_scale = delta_scale / (fpcs_new_cnt - 0.5)  # 逐步缩小范围
            a_scale_candidates = (topk_scale_candidates.unsqueeze(-1) + delta_scale_candidates.unsqueeze(-2)).reshape(*a_scale_candidates.shape[:-1], -1).clamp(min=1e-4)
            a_zero_point_candidates = topk_zp_candidates.repeat_interleave(fpcs_new_cnt, dim=-1)

            topk_index = search_strategy(self, a_scale_candidates, a_zero_point_candidates,topk=1 if remain_steps == 1 else fpcs_width)

            if remain_steps > 1:
                topk_scale_candidates = torch.gather(a_scale_candidates, dim=-1, index=topk_index)
                topk_zp_candidates = torch.gather(a_zero_point_candidates, dim=-1, index=topk_index)
            remain_steps -= 1


    def calculate_a_params(self):
        self._initialize_calib_parameters()  # 初始化
        if self.fpcs:  # 启用FPCS，搜索的时候调用本类的FPCS搜索方法，会执行多步的搜索
            # self.weight_fpcs(steps=self.steps, search_strategy=AsymmetricallyBatchingQuantLinear._search_best_w_scale_self)
            self.activation_fpcs(steps=self.steps,search_strategy=GPTQAsymmetricallyBatchingQuantLinear._search_best_a_scale_self)
        else:  # 不启用FPCS，仅搜索一步
            # weight_scale_candidates, weight_zero_point_candidates = self.calculate_percentile_weight_candidates()
            a_scale_candidates, a_zero_point_candidates = self.calculate_percentile_activation_candidates()
            # self._search_best_w_scale_self(weight_scale_candidates, weight_zero_point_candidates)
            self._search_best_a_scale_self(a_scale_candidates, a_zero_point_candidates)

        '''多轮多步搜索？'''
        for e in range(self.search_round):
            if self.fpcs:
                # self.weight_fpcs(steps=self.steps, search_strategy=AsymmetricallyBatchingQuantLinear._search_best_w_scale)
                self.activation_fpcs(steps=self.steps, search_strategy=GPTQAsymmetricallyBatchingQuantLinear._search_best_a_scale)
            else:
                # self._search_best_w_scale(weight_scale_candidates, weight_zero_point_candidates)
                self._search_best_a_scale(a_scale_candidates, a_zero_point_candidates)
        self.calibrated = True
        # del self.raw_input, self.raw_out
        return None

    # def save_hessian(self):
    #
    #     for i in range(12):
    #         file_path = os.path.join(dir_path, f'blocks.{i}.attn.qkv.hessian.pth')
    #         torch.save(self.w_quantizer.hessian,file_path)

    # def params_searching_gptq(self):

    def params_searching(self,name):
        '''这是GPTQ原始的量化参数搜索方法'''
        # self.calculate_w_params() # 这个方法仅搜索权重的量化参数——这个放这里肯定不对了。

        if self.n_V == 3:  # 这里主要是区分qkv的linear层
            self.calculate_Hessian()

            # dir_path = './test_Hessian可视化/attn.qkv'
            # os.makedirs(dir_path, exist_ok=True)
            # if 'attn.qkv' in name:
            #     file_path = os.path.join(dir_path, f'{name}.hessian.pth')
            #     torch.save(self.w_quantizer.hessian,file_path)

            with (torch.no_grad()):
                W = self.weight.data.clone()  # [576, 192]
                qkv = torch.chunk(W, 3, dim=0)  # 3 x [192, 192]
                # quantized_weight = []
                scale = []
                zero_point = []
                for weight in qkv:
                    # print(f"type(self): {type(self).__name__}")
                    # print(f"weight shape: {weight.shape}")
                    self.calculate_w_params(weight = weight) # [192,192]
                    # weight_sim = self.w_quantizer.quant(H=self.H, weight=weight, columns=self.columns, weight_flag=True)
                    scale.append(self.w_quantizer.scale)
                    zero_point.append(self.w_quantizer.zero_point)

                '''堆叠成三维张量'''
                scale = torch.stack(scale, dim=0)
                zero_point = torch.stack(zero_point, dim=0)
                self.w_quantizer.scale = nn.Parameter(scale) # [n_V,out_features//n_V,1]
                self.w_quantizer.zero_point = nn.Parameter(zero_point) # 同上
                # torch.save(self.w_quantizer.zero_point,'zero_point.pth')
                # sys.exit()
                # W_quantized = torch.cat(quantized_weight, dim=0)  # [576, 192]
            # self.quantized_weight = W_quantized
        else:
            self.calculate_Hessian()  # 计算Hessian矩阵
            with torch.no_grad():
                weight = self.weight.data.clone()
                self.calculate_w_params(weight=weight)
                # weight_sim = self.w_quantizer.quant(H=self.H, weight=weight, columns=self.columns, weight_flag=True)
                # weight_sim = self.w_quantizer(H=self.H, weight=weight, col=self.columns)
                # weight_sim = weight_sim.reshape(weight.shape)
                # self.quantized_weight = weight_sim
                self.w_quantizer.scale = nn.Parameter(self.w_quantizer.scale.unsqueeze(0)) # 同上
                self.w_quantizer.zero_point = nn.Parameter(self.w_quantizer.zero_point.unsqueeze(0)) # 同上
        # self.quantized_weight.to(self.device)
        self.calculate_a_params()  # 这个方法仅搜索激活值的量化参数
        self.calibrated = True

    """2025年8月25日：把quant_weight_bias方法和calculate_hessian方法转移到这个类中"""
    # def calculate_Hessian(self):
    #     print(f"GPTQAsymmetricallyBatchingQuantLinear类中的calculate_hessian方法的self是：{type(self).__name__}")
    #     '''这是Linear层中的计算Hessian矩阵的方法，可以进行一下简化。self是GPTQLinear'''
    #     # print(f"type of self: {type(self).__name__}")
    #     inp = self.raw_input.to(self.device).clone()
    #     # print(f"inp是否为0：{torch.equal(inp,torch.zeros_like(inp))}") # 输入不会为0.
    #     tmp = inp.shape[0]  # 拿到批次数据
    #     # 处理不同类型的输入，目的是转换成二维的矩阵
    #     if isinstance(self, nn.Linear):
    #         '''在NLP中，Linear层处理之后，这两种层的维度都是[batch, seq_len, dim(或者叫hidden_size)]'''
    #         # print(f"inp.shape: {inp.shape}")
    #         # sys.exit()
    #         if len(inp.shape) == 3:
    #             inp = inp.reshape((-1, inp.shape[-1]))  # shape:[D,B*T]，这时候每一行是一个样本
    #         if len(inp.shape) == 4:
    #             inp = inp.reshape(-1, inp.shape[-1])  # [B*H*W, C] = [25088, 384]
    #         # inp = inp.t() # 在swin transformer中，直接转置会报错，因为downsample.reduction层的输入是4维。
    #         try:
    #             inp = inp.t()  # 转置一下，shape:[B*T,D]，每一列是一个样本
    #         except:
    #             print(f"type(self) is {type(self).__name__}")
    #             print(f"inp.shape: {inp.shape}")
    #             print(f"self.raw_input.shape: {self.raw_input.shape}")
    #             print(f"self.raw_out.shape: {self.raw_out.shape}")
    #             sys.exit()
    #     # 更新Hessian矩阵，论文中的更新原理是高斯消元，但是这里好像没体现出来？
    #     self.H *= self.nsamples / (self.nsamples + tmp)  # 滑动平均 # 这里的self.H就是0
    #     self.nsamples += tmp  # 更新总样本数 # 这里就非0了
    #     inp = math.sqrt(2 / self.nsamples) * inp.float()  # 归一化缩放
    #     self.H += inp.matmul(inp.t())  # 对应论文中H=xx^T。 这里乘法用的就是数学意义上的矩阵乘，但依然能实现H的计算。
    #     print(f"计算Hessian矩阵时是否为0：{torch.equal(self.H,torch.zeros_like(self.H))}")
    #     print(f"计算Hessian矩阵计算时的结果：{self.H}")

    # def quant_weight_bias(self):
    #     print(f"调用了GPTQAsymmetricallyBatchingQuantLinear类中的quant_weight_bias方法")
    #     print(f"在GPTQAsymmetricallyBatchingQuantLinear类中quant_weight_bias方法的Hessian矩阵：{self.H.shape}")
    #     print(self.H)
    #
    #     W = self.weight.data.clone()
    #     quantized_weight = []
    #     if self.n_V == 3:
    #         scale = self.w_quantizer.scale.data.clone()
    #         zero_point = self.w_quantizer.zero_point.data.clone()
    #         with (torch.no_grad()):
    #             qkv = torch.chunk(W, self.n_V, dim=0)  # 3 x [192, 192]
    #             for i, weight in enumerate(qkv):
    #                 self.w_quantizer.scale = nn.Parameter(scale[i])
    #                 self.w_quantizer.zero_point = nn.Parameter(zero_point[i])
    #                 weight_sim = self.w_quantizer(weight, self.H, self.columns)
    #                 quantized_weight.append(weight_sim)  # 子矩阵量化，然后在拼接回去
    #             '''堆叠成三维张量'''
    #         self.w_quantizer.scale = nn.Parameter(scale)
    #         self.w_quantizer.zero_point = nn.Parameter(zero_point)
    #         weight_sim = torch.cat(quantized_weight, dim=0)  # [576, 192]
    #     else:
    #         with (torch.no_grad()):
    #             weight_sim = self.w_quantizer(self.weight, self.H, self.columns)
    #     return weight_sim, self.bias if self.bias is not None else None


class GPTQAsymmetricallyChannelWiseBatchingQuantLinear(GPTQAsymmetricallyBatchingQuantLinear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 w_bit=8,
                 a_bit=8,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 n_V=1,
                 mode="raw",
                 fpcs=False,
                 # calib_batch_size=None,
                 # search_round=1,
                 # eq_n=100,

                 # fpcs=False,
                 # steps=4
                 ):
        super().__init__(in_features, out_features, bias=bias, mode=mode, w_bit=w_bit, a_bit=a_bit, n_V=n_V,fpcs=fpcs)
        del self.a_quantizer  # 删除继承来的对称量化器，这个类应该是专门针对激活值的，权重应该还是使用的对称量化器
        self.a_quantizer = UniformQuantizer(n_bits=a_bit, symmetric=False, channel_wise=True)  # 非对称、通道级量化器
        '''这里的量化参数初始化就是通道级的量化参数'''
        self.a_quantizer.scale = nn.Parameter(torch.zeros((in_features)))
        self.a_quantizer.zero_point = nn.Parameter(torch.zeros((in_features)))
        self._prev_layer = None
        '''2025年8月4日加入fpcs搜索'''
        self.fpcs = fpcs
        self.steps = 6

    def __setattr__(self, name, value):
        '''用于自定义属性赋值行为。在这个类中，它的主要作用是安全地管理prev_layer属性的赋值，同时保持其他属性的默认赋值行为。'''
        if name == "prev_layer":  # 当检测到属性名是"prev_layer"时，直接将值存入__dict__['_prev_layer']中
            self.__dict__['_prev_layer'] = value  # 这样可以绕过Python默认的属性设置机制，避免递归调用问题
        else:  # 对于非"prev_layer"的属性，调用父类的__setattr__方法
            super().__setattr__(name, value)  # 保持所有其他属性的默认赋值逻辑不变

    @property # 将prev_layer变成一个“属性”，而不是普通的实例变量
    def prev_layer(self):
        return self._prev_layer

    @prev_layer.setter
    def prev_layer(self, layer):
        self._prev_layer = layer

    def params_searching(self,name):
        assert self.a_quantizer.channel_wise and self.w_quantizer.channel_wise  # 权重和激活必须使用通道级量化

        self._initialize_calib_parameters()  # 先初始化

        if self.fpcs:
            self.activation_fpcs(steps=self.steps,search_strategy=GPTQAsymmetricallyBatchingQuantLinear._search_best_a_scale_self)
        else:
            a_scale_candidates, a_zero_point_candidates = self.calculate_percentile_activation_candidates()
            self._search_best_a_scale_self(a_scale_candidates, a_zero_point_candidates)
        self.calibrated = True



    def reparam_step1(self):
        '''重参数化？'''
        self.calibrated = False  # 标记需要重新校准
        '''原代码这里省略了一个0，0是非对称量化的最小量化正数，这个计算公式实际上是反量化的过程，计算的结果自然就是最小值'''
        channel_min = 0 - self.a_quantizer.zero_point * self.a_quantizer.scale

        '''通过均值计算目标通道的scale和zp'''
        target_channel_scale = torch.mean(self.a_quantizer.scale).view(1)
        target_channel_zero_point = torch.mean(self.a_quantizer.zero_point).round().view(1)

        target_channel_min = -target_channel_zero_point * target_channel_scale  # 再计算一次平均的最小值
        '''r和b是lizhikai论文中公式的参数'''
        r = (self.a_quantizer.scale / target_channel_scale)  # 缩放参数
        b = channel_min / r - target_channel_min  # 偏移参数

        '''下面是对权重和偏差的调整：在当前层对上一层的权重和偏差调整，而不是论文中说的在当前层对下一层的权重和偏差进行调整'''
        self.prev_layer.weight.data = self.prev_layer.weight.data / r
        self.prev_layer.bias.data = self.prev_layer.bias.data / r.view(-1) - b
        '''这是当前层的权重'''
        self.weight.data = self.weight.data * r.view(1, -1) # 原版
        # self.quantized_weight.data = self.quantized_weight.data * r.view(1,-1)
        '''下面是当前层的bias，需要单独判断'''
        if self.bias is not None:
            self.bias.data = self.bias.data + torch.mm(self.weight.data, b.reshape(-1, 1)).reshape(-1) # 原版
            # self.bias.data = self.bias.data + torch.mm(self.quantized_weight.data, b.reshape(-1, 1)).reshape(-1)
        else:
            self.bias = nn.Parameter(torch.zeros(self.out_features))
            self.bias.data = torch.mm(self.weight.data, b.reshape(-1, 1)).reshape(-1) # 原版
            # self.bias.data = torch.mm(self.quantized_weight.data, b.reshape(-1, 1)).reshape(-1)

        return r, b, target_channel_scale, target_channel_zero_point

    def reparam(self,name):
        '''此方法和上一个reparam_step1方法都是在模型量化校准后被调用。这两个方法是lizhikai的重参数化吗？
        看方法中定义的参数名，应该是重参数化，将通道级的量化参数转换成层级的参数'''
        r, b, target_channel_scale, target_channel_zero_point = self.reparam_step1()
        # print(f"target_channel_scale: {target_channel_scale}") # 单元素张量，也可以称为标量
        # print(f"target_channel_zero_point: {target_channel_zero_point}") # 同上

        self.raw_input = (self.raw_input.cuda() / r - b).cpu()
        del self.a_quantizer.scale, self.a_quantizer.zero_point
        self.a_quantizer.channel_wise = False  # 通道级量化参数被转换了
        self.a_quantizer.scale = nn.Parameter(target_channel_scale) # 标量
        self.a_quantizer.zero_point = nn.Parameter(target_channel_zero_point)
        # sys.exit()  # 走到这里都没报错
        GPTQAsymmetricallyBatchingQuantLinear.params_searching(self,name)  # 搜索新的量化参数
        # print(f"通道级类中的hessian：{self.H.shape}，是否全0：{torch.equal(self.H, torch.zeros_like(self.H))}")
        # print(f"reparam self.a_quantizer.scale.shape = {self.a_quantizer.scale.shape}")
        # print(f"reparam self.a_quantizer.zero_point.shape = {self.a_quantizer.zero_point.shape}")
        # 66.352，65.613，66.301，66.378，65.970 AVG=66.208


class PostGeluLogBasedBatchingQuantLinear(GPTQAsymmetricallyBatchingQuantLinear):
    '''
    log_2 base = 1/k # 基于log2量化器
    k = r / q, here we set q = 37, and search the best r. # 这个是log2的底数变换
    上一个类是对GELU的双均匀量化，这个是对数量化
    '''
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 mode="raw",
                 w_bit=8,
                 a_bit=8,
                 calib_batch_size=32,
                 search_round=6,
                 eq_n=128,
                 n_V=1,
                 quantizer='adalog',  # 此类应该是支持三种量化方法：adalog、log2、logsqrt2
                 fpcs=False,
                 steps=4,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 ):
        super().__init__(in_features, out_features, bias=bias, mode=mode, w_bit=w_bit, a_bit=a_bit,
                         calib_batch_size=calib_batch_size, eq_n=eq_n, n_V=n_V, device=device)
        del self.a_quantizer
        '''这里构建一个adalog量化器'''
        self.a_quantizer = ShiftAdaLogQuantizer(n_bits=a_bit, symmetric=False, channel_wise=False)  # 这是将负值区间偏移到正数，然后统一进行量化
        self.a_quantizer.scale = nn.Parameter(torch.zeros((1)))
        self.a_quantizer.shift.data.copy_(torch.tensor(0.16997124254703522))
        self.table = torch.tensor([2 ** (-j / self.a_quantizer.r) for j in range(120)])
        self.table_scale = 1. / (4 * self.a_quantizer.n_levels - 2)
        self.table = torch.round(self.table / self.table_scale) * self.table_scale
        self.fpcs = fpcs
        self.steps = steps
        self.search_round = search_round

        if quantizer == 'log2':
            self.tmp_quantizer = ShiftLog2Quantizer(n_bits=a_bit, symmetric=False, channel_wise=False)
            self.tmp_quantizer.scale = nn.Parameter(torch.zeros((1)))
            self.tmp_quantizer.shift.data.copy_(torch.tensor(0.16997124254703522))
        elif quantizer == 'logsqrt2':
            self.tmp_quantizer = ShiftLogSqrt2Quantizer(n_bits=a_bit, symmetric=False, channel_wise=False)
            self.tmp_quantizer.scale = nn.Parameter(torch.zeros((1)))
            self.tmp_quantizer.shift.data.copy_(torch.tensor(0.16997124254703522))

        W = self.weight.data.clone()
        # W = W.flatten(1) # Linear不需要展平
        self.rows = W.shape[0]  # 行数
        self.columns = W.shape[1]  # 列数
        self.w_quantizer.hessian = torch.zeros(self.columns, self.columns).to(self.device)


    @staticmethod
    def positive_percentile(tensor, q, dim=0):
        '''
        Calculates the percentile of positive values in the tensor along a specified dimension.
        沿着张量特定的维度计算正值的百分位数
        Parameters:
        - tensor: Input tensor.
        - q: Desired percentiles, e.g., 0.5 for median.
        - dim: Dimension along which to compute the percentile.

        Output:
        The computed percentiles with shape (q.numel(), batch, height, width). # 输出张量的维度

        Note: Assumes tensor shape as (batch, channel, height, width). # 输入张量的维度
        '''
        # Create a mask for positive values
        positive_mask = tensor > 0
        # Replace negative numbers with NaN # 用nan替换非正数
        positive_tensor = torch.where(positive_mask, tensor, torch.tensor(float('nan')).to(tensor.device))
        # Sort the tensor along specified dimension # 把nan排在最后面，用于分离正数和非正数
        sorted_tensor, _ = positive_tensor.sort(dim=dim)
        # Compute the count of positive numbers
        counts = (~torch.isnan(sorted_tensor)).sum(dim=dim, keepdim=True).float()
        # Reshape q to match tensor dimensions
        shape_q = [q.numel()] + [1] * tensor.ndim
        q = q.reshape(*shape_q)
        # Calculate rank positions for desired percentiles；计算百分位数所在的位置
        ranks = ((counts * q).ceil().long() - 1).clamp(min=0)
        # Expand sorted_tensor for gathering percentiles
        expanded_sorted_tensor = sorted_tensor.unsqueeze(0).expand(q.numel(), *sorted_tensor.shape)
        # Gather the percentiles using the computed ranks
        result = torch.gather(expanded_sorted_tensor, dim + 1, ranks).squeeze(dim + 1)
        # Replace NaN values with 0
        nan_mask = torch.isnan(result)
        result.masked_fill_(nan_mask, 0)  # 如果是Nan才替换成0，不是则不需要替换
        return result

    def calculate_percentile_activation_candidates(self, l=0.9, r=1.0):
        bc = self.raw_input.shape[0]  # 初始批量大小全部设为全部校准数据
        while True:
            raw_inp = self.raw_input[:bc].cuda().view(-1)  # 取前bc个数据，展平为1D张量
            try:
                '''计算正数区间的百分位数'''
                candidates = PostGeluLogBasedBatchingQuantLinear.positive_percentile(
                    raw_inp, torch.tensor([l, r]).cuda()) + self.a_quantizer.shift.item()  # shape: (2)
                break  # 成功则只计算一次
            except:
                bc = bc // 2  # 如果批量数据太大，增加批量，减少批次数据
        candidates = candidates.unsqueeze(0)  # shape:[2]->[1,2]，2维表示左右两个边界，下面的语句就用到了
        '''scale_candidates=percentile_l+(percentile_r-percentile_l)×tensor
        这个百分位数的计算和之前的计算思路差不多，计算百分位数的差，然后再差值'''
        input_scale_candidates = candidates[:, 0:1] + (candidates[:, 1:] - candidates[:, 0:1]) * torch.tensor(
            [i / (self.eq_n - 1) for i in range(self.eq_n)]
        ).cuda().view(1, -1)
        return candidates, input_scale_candidates

    def _search_best_a_scale(self, input_scale_candidates, topk=1):
        '''在这个方法中，r固定为37，q已经搜索完成后的最优值'''
        batch_similarities = []  # similarities, need to concatenate and calculate sum (equivalent to mean with argmax)
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            x = self.raw_input[b_st:b_ed].cuda()  # 批次数据作为输入
            raw_out_expanded = self.raw_out[b_st:b_ed].cuda().unsqueeze(-2)  # shape: b,*,1,oc
            similarities = []
            for p_st in range(0, self.eq_n, self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st + self.parallel_eq_n)
                cur_a_scale = input_scale_candidates[:, p_st:p_ed]
                # quantize weight and bias；处理激活值，权重和偏差使用均匀量化器
                '''直接使用前面反量化的权重'''
                w_sim, bias_sim = self.quant_weight_bias()
                # w_sim, bias_sim = self.quant_weight_bias_for_a_params_calc()
                # w_sim = self.weight.data.clone()
                # bias_sim = self.bias.data.clone() if self.bias is not None else None
                # quantize input
                '''下面的几条语句是关于量化与反量化的，但是跟之前看到的量化语句不太一样，增加了一个偏移、缩放的步骤。还有中间过程的查表，这也是之前不常见的。'''
                x_sim = x.unsqueeze(-1)  # shape: [B,*,in_features,1]
                shifted_x_sim = ((x_sim + self.a_quantizer.shift) / cur_a_scale).clamp(min=1e-15,max=1.0)  # 偏移加放缩，限制数值范围
                x_sim_quant = torch.round(-shifted_x_sim.log2() *
                                          self.a_quantizer.r / self.a_quantizer.q)  # log2量化，但是对量化步长进行了调整。r和q是调整的超参数，如果r/q==1那这就是log2量化
                mask = x_sim_quant >= 2 * self.a_quantizer.n_levels  # 量化范围之外的值（溢出值）进行掩码？
                x_sim_quant = x_sim_quant.clamp_(0, 2 * self.a_quantizer.n_levels - 1)  # 将量化索引限制在有效范围内，避免查表时索引越界
                index = torch.remainder(x_sim_quant * self.a_quantizer.q, self.a_quantizer.r).round_().long()  # 计算查表所需的索引，用于反量化。
                x_sim = ((2 ** (-1 * torch.floor(x_sim_quant * self.a_quantizer.q / self.a_quantizer.r)))
                         * self.table.to(x.device)[index])  # 查表，反量化，将量化索引恢复为近似原始值。
                x_sim[mask] = 0  # 溢出的位置强制置0
                x_sim = (x_sim * cur_a_scale - self.a_quantizer.shift).permute(*list(range(len(x_sim.shape) - 2)), -1, -2)  # 逆缩放+逆偏移
                '''下面的语句就是计算相似度，然后再合并掉相似度'''
                out_sim = F.linear(x_sim, w_sim, bias_sim)  # shape: b,*,parallel_eq_n,out_features
                similarity = self._get_similarity(raw_out_expanded, out_sim)  # shape: b,*,parallel_eq_n,out_features
                similarity = torch.mean(similarity, dim=-1)  # shape: B,*,parallel_eq_n
                if len(similarity.shape) > 2:
                    similarity = torch.mean(similarity,dim=list(range(1, len(similarity.shape) - 1)))  # shape: b, parallel_eq_n
                similarity = torch.sum(similarity, dim=0, keepdim=True)  # shape: 1, parallel_eq_n
                similarities.append(similarity)
            # store best input scale and store in tmp_a_scale
            similarities = torch.cat(similarities, dim=1)  # shape: 1, eq_n
            batch_similarities.append(similarities)
        batch_similarities = torch.cat(batch_similarities, dim=0).sum(dim=0, keepdim=True)  # shape: 1, eq_n
        _, best_index = torch.topk(batch_similarities, k=topk, dim=-1)  # shape: 1, topk
        if topk == 1:
            tmp_a_scale = torch.gather(input_scale_candidates, dim=-1, index=best_index)
            self.a_quantizer.scale.data.copy_(tmp_a_scale.squeeze(-1))
            self.a_quantizer.update_table()
        return best_index

    def _search_best_log_base(self, q_candidates=None, topk=1):
        if q_candidates is None:
            q_candidates = torch.tensor([i for i in range(10, 11 + self.eq_n)]).cuda().view(1, -1)
        batch_similarities = []  # similarities, need to concatenate and calculate sum (equivalent to mean with argmax)
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            x = self.raw_input[b_st:b_ed].cuda()
            raw_out_expanded = self.raw_out[b_st:b_ed].cuda().unsqueeze(-2)  # shape: b,*,1,oc
            similarities = []
            for p_st in range(0, self.eq_n, self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st + self.parallel_eq_n)
                cur_q = q_candidates[:, p_st:p_ed]  # q候选值的批次数据
                # quantize weight and bias
                '''直接调用提前量化好的权重和偏差'''
                w_sim, bias_sim = self.quant_weight_bias()
                # w_sim, bias_sim = self.quant_weight_bias_for_a_params_calc()
                # w_sim = self.weight.data.clone()
                # bias_sim = self.bias.data.clone() if self.bias is not None else None
                # quantize input
                '''下面的部分好像和上面的一个方法差不多'''
                x_sim = x.unsqueeze(-1)  # shape: B,*,in_features,1
                shifted_x_sim = ((x_sim + self.a_quantizer.shift) / self.a_quantizer.scale).clamp(min=1e-15, max=1.0)
                x_sim_quant = torch.round(-shifted_x_sim.log2() * self.a_quantizer.r / cur_q)
                mask = x_sim_quant >= 2 * self.a_quantizer.n_levels
                x_sim_quant = x_sim_quant.clamp_(0, 2 * self.a_quantizer.n_levels - 1)
                '''从下面开始就不一样了，这里有一个列下标'''
                col_index = torch.remainder(x_sim_quant * cur_q, self.a_quantizer.r).round_().long()
                x_sim = (2 ** (-1 * torch.floor(x_sim_quant * cur_q / self.a_quantizer.r))) * self.table.to(x.device)[
                    col_index]
                x_sim[mask] = 0
                x_sim = (x_sim * self.a_quantizer.scale - self.a_quantizer.shift).permute(
                    *list(range(len(x_sim.shape) - 2)), -1, -2)
                ''''''
                out_sim = F.linear(x_sim, w_sim, bias_sim)  # shape: b,*,parallel_eq_n,out_features
                similarity = self._get_similarity(raw_out_expanded, out_sim)  # shape: b,*,parallel_eq_n,out_features
                similarity = torch.mean(similarity, dim=-1)  # shape: B,*,parallel_eq_n
                if len(similarity.shape) > 2:
                    similarity = torch.mean(similarity,
                                            dim=list(range(1, len(similarity.shape) - 1)))  # shape: b, parallel_eq_n
                similarity = torch.sum(similarity, dim=0, keepdim=True)  # shape: 1, parallel_eq_n
                similarities.append(similarity)
            # store best input scale and store in tmp_a_scale
            similarities = torch.cat(similarities, dim=1)  # shape: 1, eq_n
            batch_similarities.append(similarities)
        batch_similarities = torch.cat(batch_similarities, dim=0).sum(dim=0, keepdim=True)  # shape: 1, eq_n
        _, best_index = torch.topk(batch_similarities, k=topk, dim=-1)  # shape: 1, topk
        if topk == 1:
            tmp_q = torch.gather(q_candidates, dim=-1, index=best_index)
            self.a_quantizer.q.data.copy_(tmp_q.view(*self.a_quantizer.q.shape))
            self.a_quantizer.update_table()
        return best_index

    def _search_best_scale_logbase(self, input_scale_candidates, q_candidates, topk=1):
        '''根据传入的参数，这个方法应该是联合优化激活值scale和q值。
        既然能联合优化，为什么还需要前面两个单独优化的方法呢？'''
        batch_similarities = []  # similarities, need to concatenate and calculate sum (equivalent to mean with argmax)
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            x = self.raw_input[b_st:b_ed].cuda()
            raw_out_expanded = self.raw_out[b_st:b_ed].cuda().unsqueeze(-2)  # shape: b,*,1,oc
            similarities = []
            for p_st in range(0, self.eq_n, self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st + self.parallel_eq_n)
                '''同时优化q和激活值scale'''
                cur_q = q_candidates[:, p_st:p_ed]
                cur_a_scale = input_scale_candidates[:, p_st:p_ed]
                # quantize weight and bias
                w_sim, bias_sim = self.quant_weight_bias()
                # w_sim, bias_sim = self.quant_weight_bias_for_a_params_calc()
                # w_sim = self.weight.data.clone()
                # bias_sim = self.bias.data.clone() if self.bias is not None else None
                # quantize input
                x_sim = x.unsqueeze(-1)  # shape: B,*,in_features,1
                shifted_x_sim = ((x_sim + self.a_quantizer.shift) / cur_a_scale).clamp(min=1e-15, max=1.0)
                x_sim_quant = torch.round(-shifted_x_sim.log2() * self.a_quantizer.r / cur_q)
                mask = x_sim_quant >= 2 * self.a_quantizer.n_levels
                x_sim_quant = x_sim_quant.clamp_(0, 2 * self.a_quantizer.n_levels - 1)
                col_index = torch.remainder(x_sim_quant * cur_q, self.a_quantizer.r).round_().long()
                x_sim = (2 ** (-1 * torch.floor(x_sim_quant * cur_q / self.a_quantizer.r))) * self.table.to(x.device)[col_index]
                x_sim[mask] = 0
                x_sim = (x_sim * cur_a_scale - self.a_quantizer.shift).permute(*list(range(len(x_sim.shape) - 2)), -1,
                                                                               -2)
                out_sim = F.linear(x_sim, w_sim, bias_sim)  # shape: b,*,parallel_eq_n,out_features
                similarity = self._get_similarity(raw_out_expanded, out_sim)  # shape: b,*,parallel_eq_n,out_features
                similarity = torch.mean(similarity, dim=-1)  # shape: B,*,parallel_eq_n
                if len(similarity.shape) > 2:
                    similarity = torch.mean(similarity, dim=list(range(1, len(similarity.shape) - 1)))  # shape: b, parallel_eq_n
                similarity = torch.sum(similarity, dim=0, keepdim=True)  # shape: 1, parallel_eq_n
                similarities.append(similarity)
            # store best input scale and store in tmp_a_scale
            similarities = torch.cat(similarities, dim=1)  # shape: 1, eq_n
            batch_similarities.append(similarities)
        batch_similarities = torch.cat(batch_similarities, dim=0).sum(dim=0, keepdim=True)  # shape: 1, eq_n
        _, best_index = torch.topk(batch_similarities, k=topk, dim=-1)  # shape: 1, topk
        if topk == 1:
            tmp_a_scale = torch.gather(input_scale_candidates, dim=-1, index=best_index)
            tmp_q = torch.gather(q_candidates, dim=-1, index=best_index)
            self.a_quantizer.scale.data.copy_(tmp_a_scale.squeeze(-1))
            self.a_quantizer.q.data.copy_(tmp_q.view(*self.a_quantizer.q.shape))
            self.a_quantizer.update_table()
        return best_index

    def activation_fpcs(self, ud_candidates, base_num=8, scale_num=16, fpcs_width=32, steps=6):
        '''
        ud_candidates：好像是通过百分位数计算出来的候选值，但是是哪个值的候选值？上下界？
        '''
        fpcs_new_cnt = int(self.eq_n / fpcs_width)
        '''q的取值，不能太大，也不能太小，这是因为log的底数k=r/q，r固定为37，q过大或者过小都会影响底数k，从而导致量化误差的增加
        从10开始是一个经验值，通过实验得出，暂时不明白为什么要这么做。2025年6月6日'''
        q_candidates = torch.tensor([i for i in range(10, 11 + self.eq_n)]).cuda().view(1, -1)  # 生成候选值，尽可能覆盖广阔的区间
        q_best_index = self._search_best_log_base(q_candidates, topk=base_num)  # 初步筛选
        '''在ud_candidates给出的上下界间生成scale_num个线性间隔的scale候选。'''
        a_scale_candidates = ud_candidates[:, 0:1] + (ud_candidates[:, 1:] - ud_candidates[:, 0:1]) * torch.tensor(
            [i / (scale_num - 1) for i in range(scale_num)]
        ).cuda().view(1, -1)
        delta_scale = a_scale_candidates[:, 1:2] - a_scale_candidates[:, 0:1]  # a_scale_candidates就两个维度，1:2和1:是同样的作用
        '''扩展候选参数，配对成(q, scale)形式的候选集，也就是q_candidates'''
        a_scale_candidates = a_scale_candidates.repeat(1, base_num)  #
        q_candidates = torch.gather(q_candidates, dim=-1, index=q_best_index)
        q_candidates = q_candidates.repeat_interleave(scale_num, dim=-1)
        '''上面配对完成后，以(q, scale)的形式联合搜索最优的参数组合'''
        topk_index = self._search_best_scale_logbase(a_scale_candidates, q_candidates, topk=fpcs_width)
        topk_scale_candidates = torch.gather(a_scale_candidates, dim=-1, index=topk_index)
        topk_q_candidates = torch.gather(q_candidates, dim=-1, index=topk_index)
        '''上面的过程相当于初始化，下面开始渐进搜索'''
        remain_steps = steps - 1
        while remain_steps > 0:
            delta_scale_candidates = (torch.linspace(0, 1, steps=fpcs_new_cnt).cuda()[None, :] - 0.5) * delta_scale
            delta_scale = delta_scale / (fpcs_new_cnt - 0.5)
            a_scale_candidates = (topk_scale_candidates.unsqueeze(-1) + delta_scale_candidates.unsqueeze(-2)).reshape(
                *a_scale_candidates.shape[:-1], -1)
            q_candidates = topk_q_candidates.repeat_interleave(fpcs_new_cnt, dim=-1)
            topk_index = self._search_best_scale_logbase(a_scale_candidates, q_candidates,
                                                         topk=1 if remain_steps == 1 else fpcs_width)
            if remain_steps > 1:
                topk_scale_candidates = torch.gather(a_scale_candidates, dim=-1, index=topk_index)
                topk_q_candidates = torch.gather(q_candidates, dim=-1, index=topk_index)
            remain_steps -= 1

    def calculate_a_params(self):
        self._initialize_calib_parameters()
        ud_candidates, input_scale_candidates = self.calculate_percentile_activation_candidates()
        self.a_quantizer.scale.data.copy_(input_scale_candidates[:, -2])
        self.a_quantizer.inited = True

        '''上面的一部分是进行初始化或者叫第一轮搜索，下面的代码开始多轮搜索'''
        for e in range(self.search_round):
            if self.fpcs:
                self.activation_fpcs(ud_candidates=ud_candidates, steps=self.steps)
            else:  # 不启用FPCS，通过百分位数进行计算
                self._search_best_log_base()
                self._search_best_a_scale(input_scale_candidates)
                # weight_scale_candidates, weight_zero_point_candidates = self.calculate_percentile_weight_candidates()
                # self._search_best_w_scale(weight_scale_candidates, weight_zero_point_candidates)
        '''如果存在临时量化器，则进行切换。这是为了保证在搜索阶段能找到更优的参数，在部署阶段又能兼顾效率。'''
        if hasattr(self, 'tmp_quantizer'):
            self.tmp_quantizer.scale.data.copy_(self.a_quantizer.scale.data)  # 当前量化器的参数复制到临时量化器
            self.tmp_quantizer.inited = True
            self.a_quantizer = self.tmp_quantizer  # 将主量化器替换为临时量化器
            del self.tmp_quantizer  # 删除临时量化器

        self.calibrated = True
        # del self.raw_input, self.raw_out

    # 原版的参数搜索，会导致模型推理准确率为0。
    def params_searching_origin(self): # 这个方法写的不是很行，直接从父类里面复制过来的，没有针对mlp.fc2进行优化
        self.calculate_a_params()
        if self.n_V == 3: # 这里不走这个if分支，走的是else分支
            self.calculate_Hessian()  # 如果为每个子矩阵都计算Hessian矩阵，理论上输入不变Hessian矩阵也是不变的，但是这里发生改变了，说明输入值也被改变了。
            with torch.no_grad():
                W = self.weight.data.clone()  # [576, 192]
                qkv = torch.chunk(W, 3, dim=0)  # 3 x [192, 192]
                quantized_weight = []
                for weight in qkv:
                    '''参考Conv2D中的代码修改下面的方法'''
                    weight_sim = self.w_quantizer.quant(H=self.H, weight=weight, columns=self.columns, weight_flag=True)  # 这里面还是修改Hessian矩阵了，但是应该是不能修改的
                    quantized_weight.append(weight_sim)
                W_quantized = torch.cat(quantized_weight, dim=0)  # [576, 192]
            self.quantized_weight = W_quantized
        else:
            self.calculate_Hessian()  # 计算Hessian矩阵
            '''这里传入的权重数据一定是卷积层的权重，所以，只要把quant()方法中的判断语句拿到外面来即可。'''
            weight = self.weight.data.clone()
            weight_sim = self.w_quantizer.quant(H=self.H, weight=weight, columns=self.columns, weight_flag=True)  # 直接量化权重，这个方法里面包含了初始化量化参数的部分。
            weight = weight_sim.reshape(weight.shape)
            self.quantized_weight = weight
        self.quantized_weight.to(self.device)
        self.calibrated = True
    '''2025年8月21日：重写上面的参数搜索方法，使其作用仅仅是搜索量化参数，不涉及量化'''
    def params_searching(self,name):

        self.calculate_Hessian()
        with torch.no_grad():
            weight = self.weight.data.clone()
            self.calculate_w_params(weight = weight)
            self.w_quantizer.scale = nn.Parameter(self.w_quantizer.scale.unsqueeze(0))  # 同上
            self.w_quantizer.zero_point = nn.Parameter(self.w_quantizer.zero_point.unsqueeze(0))  # 同上
        self.calculate_a_params()
        self.calibrated = True


    """以下的代码是2025年8月20日新增的部分，主要是补上mlp.fc2这一层权重量化参数的搜索"""
    def _search_best_log_base(self, q_candidates=None, topk=1):
        '''首先明白这个方法要干什么，要从q的候选值里面搜索最优的log底数（r是固定的37）'''
        if q_candidates is None:
            q_candidates = torch.tensor([i for i in range(10, 11 + self.eq_n)]).cuda().view(1, -1)
        batch_similarities = [] # similarities, need to concatenate and calculate sum (equivalent to mean with argmax)
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            x = self.raw_input[b_st:b_ed].cuda()
            raw_out_expanded = self.raw_out[b_st:b_ed].cuda().unsqueeze(-2) # shape: b,*,1,oc
            similarities = []
            for p_st in range(0,self.eq_n,self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st+self.parallel_eq_n)
                cur_q = q_candidates[:, p_st:p_ed] # q候选值的批次数据
                # quantize weight and bias
                w_sim, bias_sim = self.quant_weight_bias()
                # w_sim, bias_sim = self.quant_weight_bias_for_a_params_calc()
                # w_sim = self.weight.data.clone()
                # bias_sim = self.bias.data.clone() if self.bias is not None else None
                # quantize input
                '''下面的部分好像和上面的一个方法差不多'''
                x_sim = x.unsqueeze(-1) # shape: B,*,in_features,1
                shifted_x_sim = ((x_sim + self.a_quantizer.shift) / self.a_quantizer.scale).clamp(min=1e-15, max=1.0)
                x_sim_quant = torch.round(-shifted_x_sim.log2() * self.a_quantizer.r / cur_q)
                mask = x_sim_quant >= 2 * self.a_quantizer.n_levels
                x_sim_quant = x_sim_quant.clamp_(0, 2 * self.a_quantizer.n_levels - 1)
                '''从下面开始就不一样了，这里有一个列下标'''
                col_index = torch.remainder(x_sim_quant * cur_q, self.a_quantizer.r).round_().long()
                x_sim = (2 ** (-1 * torch.floor(x_sim_quant * cur_q / self.a_quantizer.r))) * self.table.to(x.device)[col_index]
                x_sim[mask] = 0
                x_sim = (x_sim * self.a_quantizer.scale - self.a_quantizer.shift).permute(*list(range(len(x_sim.shape)-2)),-1,-2)
                ''''''
                out_sim = F.linear(x_sim, w_sim, bias_sim) # shape: b,*,parallel_eq_n,out_features
                similarity = self._get_similarity(raw_out_expanded, out_sim) # shape: b,*,parallel_eq_n,out_features
                similarity = torch.mean(similarity, dim=-1) # shape: B,*,parallel_eq_n
                if len(similarity.shape) > 2:
                    similarity = torch.mean(similarity, dim=list(range(1,len(similarity.shape)-1))) # shape: b, parallel_eq_n
                similarity = torch.sum(similarity, dim=0, keepdim=True) # shape: 1, parallel_eq_n
                similarities.append(similarity)
            # store best input scale and store in tmp_a_scale
            similarities = torch.cat(similarities, dim=1) # shape: 1, eq_n
            batch_similarities.append(similarities)
        batch_similarities = torch.cat(batch_similarities, dim=0).sum(dim=0, keepdim=True) # shape: 1, eq_n
        _, best_index = torch.topk(batch_similarities, k=topk, dim=-1) # shape: 1, topk
        if topk == 1:
            tmp_q = torch.gather(q_candidates, dim=-1, index=best_index)
            self.a_quantizer.q.data.copy_(tmp_q.view(*self.a_quantizer.q.shape))
            self.a_quantizer.update_table()
        return best_index

    # def params_searching_gptq(self):
    # # def params_searching(self):
    #     '''这是GPTQ原始的量化参数搜索方法'''
    #     self.calculate_a_params()  # 这个方法仅搜索激活值的量化参数
    #     self.calculate_Hessian()  # 计算Hessian矩阵
    #     with torch.no_grad():
    #         weight = self.weight.data.clone()
    #         weight_sim = self.w_quantizer.quant(H=self.H, weight=weight, columns=self.columns, weight_flag=True)
    #         weight_sim = weight_sim.reshape(weight.shape)
    #         self.quantized_weight = weight_sim
    #         self.w_quantizer.scale = nn.Parameter(self.w_quantizer.scale.unsqueeze(0))
    #         self.w_quantizer.zero_point = nn.Parameter(self.w_quantizer.zero_point.unsqueeze(0))
    #     self.quantized_weight.to(self.device)
    #     self.calibrated = True
    """2025年8月20日新增部分结束"""

    def reparam_bias(self):
        '''偏置重参数化？'''
        if self.a_quantizer.bias_reparamed:
            return
        # 用于创建一个形状指定且所有元素值相同的张量，应该是用来抵消原来的偏移量？
        x_ = torch.full((1, self.in_features), -self.a_quantizer.shift.item()).cuda()
        '''下面的两条语句是后来修改的'''
        w_sim, bias_sim = self.quant_weight_bias()
        # w_sim, bias_sim = self.quant_weight_bias_for_a_params_calc()
        # w_sim = self.weight.data.clone()
        # bias_sim = self.bias if self.bias is not None else torch.zeros_like(w_sim)
        x_ = (x_ @ w_sim.transpose(0, 1)).squeeze()  # 计算偏移量的贡献？
        self.bias.data.copy_(bias_sim + x_)
        self.a_quantizer.bias_reparamed.data.copy_(torch.tensor(True))
    # 66.403，66.327，66.276，66.378，65.511 AVG=66.327