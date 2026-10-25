import sys
from types import MethodType
import torch
import torch.nn.functional as F
import timm
from timm.models import vision_transformer
from timm.models.vision_transformer import Attention
from timm.models.swin_transformer import WindowAttention
import torch.nn as nn

from quant_layers import MinMaxQuantMatMul
from utils.models import vit_attn_forward,swin_attn_forward,MatMul
import re
from quant_layers.conv import GPTQConv2D
from quant_layers.linear import GPTQLinear, GPTQAsymmetricallyBatchingQuantLinear,GPTQAsymmetricallyChannelWiseBatchingQuantLinear
from quant_layers.linear import PostGeluLogBasedBatchingQuantLinear
from quant_layers.matmul import AsymmetricallyBatchingQuantMatMul,PostSoftmaxAsymmetricallyBatchingQuantMatMul

def wrap_modules_in_net(model, cfg, reparam = False):
    '''将Conv、Linear和Matmul替换成支持量化的层。'''
    module_dict={}
    for name, module in model.named_modules():
        module_dict[name] = module
        idx = name.rfind('.')
        if idx == -1:
            idx = 0
        father_name = name[:idx]
        if father_name in module_dict:
            father_module = module_dict[father_name]
        else:
            raise RuntimeError(f"father module {father_name} not found")

        '''卷积层的包装'''
        # if isinstance(module,nn.Conv2d):
        #     # print(f'module type is {type(module)}') # module是nn.Conv2D，不是包装完的GPTQConv2D
        #     idx = idx + 1 if idx != 0 else idx # 之前把这行代码漏了，怪不得老是报错。
        #     new_module = GPTQConv2D(
        #         in_channels=module.in_channels,
        #         out_channels=module.out_channels,
        #         kernel_size=module.kernel_size,
        #         stride=module.stride,
        #         padding=module.padding,
        #         dilation=module.dilation,
        #         groups=module.groups,
        #         bias=True,
        #         padding_mode=module.padding_mode,
        #         channel_wise=True,
        #         symmetric=False,
        #         mse=True,
        #         w_bit=cfg.w_bit,
        #         a_bit=cfg.qconv_a_bit,
        #         blocksize=128,
        #         percdamp=0.01,
        #         groupsize=-1,
        #         actorder=False,
        #         static_groups=False,
        #         device=next(module.parameters()).device, # 原版的conv2d没有这个属性
        #         mode='raw',
        #         calib_batch_size = cfg.calib_batch_size,
        #         eq_n = cfg.eq_n,
        #         search_round = cfg.search_round,
        #         fpcs=cfg.fpcs,
        #         steps=cfg.steps,
        #     )
        #     new_module.weight.data.copy_(module.weight.data)
        #     new_module.bias.data.copy_(module.bias)
        #     setattr(father_module, name[idx:], new_module)


        '''Linear层的包装，所有的Linear层都包装'''
        if isinstance(module, nn.Linear):
            '''所有的Linear层都会进入这个分支'''
            cur_a_bit = cfg.qhead_a_bit if 'head' in name else cfg.a_bit
            linear_kwargs = { # 这个键值对的参数太多了，应该进行一些简化，只保留Linear层基础的参数。
                'in_features': module.in_features,
                'out_features': module.out_features,
                'bias': True,
                'device': next(module.parameters()).device,
                'n_V': 3 if 'qkv' in name else 1,  # 这里得用父类转字符串才能拿到？
                'mode': 'raw',
                'w_bit':cfg.w_bit,
                'a_bit':cur_a_bit,
                'fpcs' : cfg.fpcs,
                # 'channel_wise':True, # 是否采用通道级量化不应该在这里判断，而是针对不同的层进行单独判断
                # 'symmetric':False,
                # 'blocksize':128,
                # 'percdamp':0.01,
                # 'groupsize':-1,
                # 'actorder':False,
                # 'static_groups':True,
                }
            idx = idx + 1 if idx != 0 else idx
            '''下面的这个if判断的并不完整，如果后面出现问题再去看AdaLog怎么判断的。'''
            if reparam and ( 'qkv' in name or 'fc1' in name or 'reduction' in name):
                idxx = father_name.rfind('.')
                idxx = 0 if idxx == -1 else idxx
                grandfather_name = father_name[:idxx]
                if grandfather_name in module_dict:
                    grandfather_module = module_dict[grandfather_name]
                new_module = GPTQAsymmetricallyChannelWiseBatchingQuantLinear(  # 这里是非对称、通道级量化
                    **linear_kwargs,
                )
                if 'qkv' in name:
                    new_module.prev_layer = grandfather_module.norm1
                if 'fc1' in name:
                    new_module.prev_layer = grandfather_module.norm2
                if 'reduction' in name:
                    new_module.prev_layer = father_module.norm
            elif 'fc2' in name:
                # print(f"包装mlp.fc2")
                # 写完GELU的量化再写这个分支的语句。但是有个问题，如果这里对fc2进行操作了，那么下面的else分支就会排除掉fc2，可能会有问题，所以还是直接注释掉吧。
                # new_module = module # mlp.fc2这个Linear层不进行任何操作。
                new_module = PostGeluLogBasedBatchingQuantLinear(  # 激活值的量化器
                    **linear_kwargs,
                    quantizer='adalog',
                )
            else:
                new_module = GPTQAsymmetricallyBatchingQuantLinear(**linear_kwargs)

            new_module.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                new_module.bias.data.copy_(module.bias.data)
            setattr(father_module, name[idx:], new_module)

        '''Linear层包装，只对一种类型的Linear层进行包装，用于Debug'''
        # if isinstance(module, nn.Linear):
        #     '''所有的Linear层都会进入这个分支'''
        #     cur_a_bit = cfg.qhead_a_bit if 'head' in name else cfg.a_bit
        #     linear_kwargs = { # 这个键值对的参数太多了，应该进行一些简化，只保留Linear层基础的参数。
        #         'in_features': module.in_features,
        #         'out_features': module.out_features,
        #         'bias': True,
        #         'device': next(module.parameters()).device,
        #         'n_V': 3 if 'qkv' in name else 1,  # 这里得用父类转字符串才能拿到？
        #         'mode': 'raw',
        #         'w_bit':cfg.w_bit,
        #         'a_bit':cur_a_bit,
        #         'fpcs' : cfg.fpcs,
        #     }
        #     idx = idx + 1 if idx != 0 else idx
        #     '''下面的这个if判断的并不完整，如果后面出现问题再去看AdaLog怎么判断的。'''
        #     # if 'qkv' in name or 'fc1' in name or 'reduction' in name:
        #     # if 'qkv' in name:
        #     #     idxx = father_name.rfind('.')
        #     #     idxx = 0 if idxx == -1 else idxx
        #     #     grandfather_name = father_name[:idxx]
        #     #     if grandfather_name in module_dict:
        #     #         grandfather_module = module_dict[grandfather_name]
        #     #     new_module = GPTQAsymmetricallyChannelWiseBatchingQuantLinear(  # 这里是非对称、通道级量化
        #     #         **linear_kwargs,
        #     #     )
        #     #     if 'qkv' in name:
        #     #         new_module.prev_layer = grandfather_module.norm1
        #     #     if 'fc1' in name:
        #     #         new_module.prev_layer = grandfather_module.norm2
        #     # if 'fc2' in name:
        #     #     new_module = PostGeluLogBasedBatchingQuantLinear(  # 激活值的量化器
        #     #         **linear_kwargs,
        #     #         quantizer='adalog',
        #     #     )
        #     # else: # 把这个else分支注释掉不久排除掉fc2了吗？
        #     #     '''如果直接注释掉这个else分支，会导致维度不匹配的报错'''
        #     #     new_module = GPTQAsymmetricallyBatchingQuantLinear(**linear_kwargs)
        #     # elif 'head' in name:
        #     #     new_module = GPTQAsymmetricallyBatchingQuantLinear(**linear_kwargs)
        #     if 'proj' in name:
        #         new_module = GPTQAsymmetricallyBatchingQuantLinear(**linear_kwargs)
        #     else: # 其余层不包装，就用原来的module
        #         new_module = module
        #     new_module.weight.data.copy_(module.weight.data)
        #     if module.bias is not None:
        #         new_module.bias.data.copy_(module.bias.data)
        #     setattr(father_module, name[idx:], new_module)

        '''矩阵乘法的包装'''
        if isinstance(module, MatMul):
            idx = idx + 1 if idx != 0 else idx
            matmul_kwargs = {
                'B_bit': cfg.a_bit,
                'mode': 'raw',
                'calib_batch_size': cfg.calib_batch_size,
                'search_round': cfg.search_round,
                'eq_n': cfg.eq_n,
                'head_channel_wise': cfg.matmul_head_channel_wise,
                'num_heads': father_module.num_heads,
                # 这个num_heads也是会根据模型的不同发生变化的，DeiT-Tiny中heads=3，而ViT-Base网络则是heads=12
                'fpcs': cfg.fpcs,
                'steps': cfg.steps,
            }
            if 'matmul2' in name: # 这个是softmax和V矩阵相乘
                new_module = PostSoftmaxAsymmetricallyBatchingQuantMatMul(
                    A_bit = cfg.s_bit,
                    **matmul_kwargs,
                    quantizer = cfg.post_softmax_quantizer,
                )
            else: # matmul1：QK矩阵相乘
                new_module = AsymmetricallyBatchingQuantMatMul(
                    A_bit = cfg.a_bit,
                    **matmul_kwargs
                )
            setattr(father_module, name[idx:], new_module)

        '''Linear层的包装——用于测试swin transformer'''
        # if isinstance(module, nn.Linear):
        #     linear_kwargs = { # 这个键值对的参数太多了，应该进行一些简化，只保留Linear层基础的参数。
        #         'in_features': module.in_features,
        #         'out_features': module.out_features,
        #         'bias': True,
        #         'device': next(module.parameters()).device,
        #         'n_V': 3 if 'qkv' in name else 1,  # 这里得用父类转字符串才能拿到？
        #         'mode': 'raw',
        #         'w_bit':4,
        #         'a_bit':4,
        #         'fpcs' : True,
        #     }
        #     idx = idx + 1 if idx != 0 else idx
        #     '''下面的这个if判断的并不完整，如果后面出现问题再去看AdaLog怎么判断的。'''
        #     if 'reduction' in name: # reduction是Swin T中的层，属于是战未来。
        #         idxx = father_name.rfind('.')
        #         idxx = 0 if idxx == -1 else idxx
        #         grandfather_name = father_name[:idxx]
        #         if grandfather_name in module_dict:
        #             grandfather_module = module_dict[grandfather_name]
        #         new_module = GPTQAsymmetricallyChannelWiseBatchingQuantLinear(  # 这里是非对称、通道级量化
        #             **linear_kwargs,
        #         )
        #         if 'reduction' in name:
        #             new_module.prev_layer = father_module.norm
        #     else:
        #         new_module = module
        #     new_module.weight.data.copy_(module.weight.data)
        #     if module.bias is not None:
        #         new_module.bias.data.copy_(module.bias.data)
        #     setattr(father_module, name[idx:], new_module)
    return model

def wrap_reparamed_modules_in_net(model):
    '''这个方法的作用是将AsymmetricallyChannelWiseBatchingQuantLinear这个通道级量化的类，重新包装成AsymmetricallyBatchingQuantLinear非通道级量化的类。'''
    module_dict={}
    for name, module in model.named_modules():
        module_dict[name] = module
        idx = name.rfind('.')
        if idx == -1:
            idx = 0
        father_name = name[:idx]
        if father_name in module_dict:
            father_module = module_dict[father_name]
        else:
            raise RuntimeError(f"father module {father_name} not found")

        if isinstance(module, GPTQAsymmetricallyChannelWiseBatchingQuantLinear):
            idx = idx + 1 if idx != 0 else idx
            linear_kwargs = {
                'in_features': module.in_features,
                'out_features': module.out_features,
                'bias': module.bias is not None,
                'mode': module.mode,
                'w_bit': module.w_quantizer.n_bits,
                'a_bit': module.a_quantizer.n_bits,
                'calib_batch_size': module.calib_batch_size,
                # 'search_round': module.search_round,
                'eq_n': module.eq_n,
                'n_V': module.n_V,
                # 'fpcs': module.fpcs,
                # 'steps': module.steps,
            }
            new_module = GPTQAsymmetricallyBatchingQuantLinear(**linear_kwargs)
            # print(f"module.w_quantizer.scale.shape = {module.w_quantizer.scale.shape}")
            # print(f"module.w_quantizer.zero_point.shape = {module.w_quantizer.zero_point.shape}")
            # print(f"new_module.w_quantizer.scale.shape = {new_module.w_quantizer.scale.shape}")
            # print(f"new_module.w_quantizer.zero_point.shape = {new_module.w_quantizer.zero_point.shape}")
            # print(f"module.state_dict() = {module.state_dict().keys()}")
            # print(f"new_module.state_dict() = {new_module.state_dict().keys()}")
            new_module.load_state_dict(module.state_dict())
            new_module.w_quantizer.hessian = module.w_quantizer.hessian # 重新把Hessian矩阵拿过来。
            new_module.w_quantizer.cols = module.w_quantizer.cols # 重新把Hessian矩阵拿过来。
            # new_module.w_quantizer.rest = module.w_quantizer.rest # 不应该在这里收集舍入值。
            new_module.calibrated = True
            new_module.a_quantizer.inited = True
            new_module.w_quantizer.inited = True
            setattr(father_module, name[idx:], new_module)
    return model



""" 下面的这个包装方法好像是PTQ4ViT中的，还是用AdaLog的包装方法吧。
def wrap_modules_in_net(net,cfg):
    wrapped_modules={} # 字典，用于存储替换后的模块（即量化后的模块）。
    module_dict={} # 字典，用于存储模型中各个层的引用。
    module_types = {"qkv":"qlinear_qkv", "proj":'qlinear_proj', 'fc1':'qlinear_MLP_1', 'fc2':'qlinear_MLP_2', 'head':'qlinear_classifier',
                    'matmul1':"qmatmul_qk", 'matmul2':"qmatmul_scorev", "reduction": "qlinear_reduction", 'fc':'qlinear_classifier' }
    '''这几行代码到底是从哪里抄过来的？
    # 上面一行缺少对‘fc’的映射，导致报错。不知道将‘fc’映射成'fc1'的'qlinear_MLP_1'还是'fc2'的'qlinear_MLP_2'，再或者是'head'的'qlinear_classifier'
    # 这个fc只出现在最后的一个分类头（ClassifierHead）中，应该遵循head的量化方法，理想的结果是量化完head之后，head的结构改变，没有fc，也就不需要再对fc进行量化了？
    '''
    # 字典，定义了不同模块的类型映射。
    # 把网络中FP32的层替换成全精度的层
    it=[(name,m) for name,m in net.named_modules()] # 返回模型中所有子模块的名称和模块对象。it是一个包含（模块名, 模块）元组的列表。
    # 这里的模块已经不是初始的模型结构了,加入了一些其他的层.
    for name,m in it: # name对应的是一个str,表示模块名,m对应的是具体的模块,这个模块是torch.nn提供的模块.
        module_dict[name]=m
        idx=name.rfind('.') # 返回字符串'.'最后一次出现的位置，如果没有匹配项则返回-1。相当于搜索子模块吗?
        if idx==-1:
            idx=0
        father_name=name[:idx]
        if father_name in module_dict:
            father_module=module_dict[father_name]
        else:
            raise RuntimeError(f"father module {father_name} not found")
        if isinstance(m,nn.Conv2d): # 匹配到torch.nn.Conv2d之后,进行替换
            # Embedding Layer
            idx = idx+1 if idx != 0 else idx
            new_m=cfg.get_module("qconv",m.in_channels,m.out_channels,m.kernel_size,m.stride,m.padding,m.dilation,m.groups,m.bias is not None,m.padding_mode)
            # get_module方法在configs目录下的两个配置文件中，根据不同的配置文件选择获取module的方法
            new_m.weight.data=m.weight.data
            new_m.bias=m.bias
            replace_m=new_m
            wrapped_modules[name] = new_m
            setattr(father_module,name[idx:],replace_m)
            # setattr(object, name, value) : object 是要设置属性的对象;name 是属性的名称;value 是要设置的属性值。
        elif isinstance(m,nn.Linear): # 线性层,attn.qkv属于linear
            # Linear Layer
            idx = idx+1 if idx != 0 else idx
            new_m = cfg.get_module(module_types[name[idx:]],m.in_features,m.out_features)
            new_m.weight.data=m.weight.data
            new_m.bias=m.bias
            replace_m=new_m
            wrapped_modules[name] = new_m
            setattr(father_module,name[idx:],replace_m)
        elif isinstance(m,MatMul): # 矩阵乘
            # Matmul Layer
            idx = idx+1 if idx != 0 else idx
            new_m = cfg.get_module(module_types[name[idx:]])
            replace_m=new_m
            wrapped_modules[name] = new_m
            setattr(father_module,name[idx:],replace_m)
    print("Completed net wrap.")
    return wrapped_modules
"""