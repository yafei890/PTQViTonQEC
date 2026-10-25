import sys
import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from quant_layers import MinMaxQuantMatMul,MinMaxQuantConv2d,MinMaxQuantLinear


class QuantCalibrator:
    def __init__(self, model, calib_loader):
        self.model = model.to('cuda')
        self.calib_loader = calib_loader

    def single_input_forward_hook(self, module, inp, outp):
        # print(f"type of module: {type(module).__name__}")
        if module.tmp_input is None:
            module.tmp_input = []
        module.tmp_input.append(inp[0].cpu().detach())
        
    def double_input_forward_hook(self, module, inp, outp):
        # print(f"type of self : {type(self)}")
        # print(f"type of module : {type(module)}")
        if module.tmp_input is None:
            module.tmp_input = [[],[]]
        module.tmp_input[0].append(inp[0].cpu().detach())
        module.tmp_input[1].append(inp[1].cpu().detach())
    
    def outp_forward_hook(self, module, inp, outp):
        # print(f"type of module : {type(module)}")
        if module.tmp_out is None:
            module.tmp_out = []
        module.tmp_out.append(outp.cpu().detach())

    def batching_quant_calib(self):
        device = next(self.model.parameters()).device
        total = sum(1 for name, module in self.model.named_modules() if hasattr(module, 'calibrated') and not module.calibrated) # 计算需要校准的模块总数
        with tqdm(total=total) as progress_bar: # 进度条
            for name, module in self.model.named_modules():
                if not hasattr(module, 'calibrated') or module.calibrated: # 没有calibrated属性或者已校准
                    continue # 跳过不需要校准的层
                progress_bar.set_description(f"calibrating {name}")
                hooks = [] # 注册hook，会监听该模块完整的前向计算过程，包括输入和输出
                hooks.append(module.register_forward_hook(self.outp_forward_hook))


                # if isinstance(module, GPTQLinear) or isinstance(module, GPTQConv2D): # 线性层和卷积层采用相同的处理方法，因为这两个都是单项输入
                #     hooks.append(module.register_forward_hook(self.single_input_forward_hook))
                if isinstance(module, MinMaxQuantConv2d): # 线性层和卷积层采用相同的处理方法，因为这两个都是单项输入
                    hooks.append(module.register_forward_hook(self.single_input_forward_hook))
                if isinstance(module, MinMaxQuantLinear):  # 线性层和卷积层采用相同的处理方法，因为这两个都是单项输入
                    hooks.append(module.register_forward_hook(self.single_input_forward_hook))
                if isinstance(module, MinMaxQuantMatMul): # 矩阵乘是两个输入，注册双项的钩子
                    hooks.append(module.register_forward_hook(self.double_input_forward_hook))



                with torch.no_grad():
                    for i, (inp, target) in enumerate(self.calib_loader): # 循环整个校准集
                        # print(f"i = {i}") # i一直为0，也就是也就是一个批次就加载完了。
                        # print(f"校准时的输入数据：{inp.shape}")
                        # print(f"校准时的输入：{inp}")
                        inp = inp.to(device)
                        # torch.save(inp,'inp.pt')
                        _ = self.model(inp) # 模型的输出结果不重要，只是要通过hook搜集中间的数据，存放在tmp_input和tmp_output
                        # print("执行：self.model(inp)")

                """上面就是用模型跑一遍校准数据，拿到输入的值"""

                # replace cached raw_inputs, raw_outs 将收集的数据拼接成完整张量
                try:
                    module.raw_out = torch.cat(module.tmp_out, dim=0)
                except:
                    print(f"type of module : {type(module)}")
                    print(f"module.tmp_out : {module.tmp_out}")
                    # sys.exit()
                module.nsamples=len(self.calib_loader.dataset)
                # if isinstance(module, GPTQLinear) or isinstance(module, GPTQConv2D):
                #     module.raw_input = torch.cat(module.tmp_input, dim=0)
                if isinstance(module, MinMaxQuantConv2d): # 这两个if可以或在一起，但是拆开是为了直到哪里报错
                    module.raw_input = torch.cat(module.tmp_input, dim=0)
                if isinstance(module, MinMaxQuantLinear): # 2025年7月23日14点44分：报错，和昨天Conv层的报错一样，都是什么“NoneType”
                    # 确定是Conv层的报错吗？没有input不就说明上一层没有正确输出吗？
                    module.raw_input = torch.cat(module.tmp_input, dim=0)
                '''对于矩阵乘法，暂时不需要这一步，因为我用的MinMax量化方法，量化参数的计算很简单，用的时候直接算，不需要提前搜索'''
                if isinstance(module, MinMaxQuantMatMul):
                    module.raw_input = [torch.cat(_, dim=0) for _ in module.tmp_input]
                # if isinstance(module, MinMaxQuantMatMul):
                #     module.raw_input = [torch.cat(_, dim=0) for _ in module.tmp_input]
                # if isinstance(module, GPTQConv2D):
                #     # print("往self.input中写入数据......")
                #     module.raw_input = torch.cat(module.tmp_input, dim=0)
                for hook in hooks:
                    hook.remove() # 收集完数据，清除hook
                module.tmp_input = module.tmp_out = None # 清空缓存？
                # run hyperparameter_searching
                with torch.no_grad():
                    # module.quant_forward() # 量化？
                    # with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
                    module.params_searching(name) # 必须保证Conv2D、Linear和MatMul都有这个方法
                    # prof.export_chrome_trace('./profile.json')
                    if hasattr(module, 'prev_layer') and module.prev_layer is not None:
                        progress_bar.set_description(f"reparaming {name}")
                        module.reparam(name)
                progress_bar.update()


        # end calibration
        for name, module in self.model.named_modules():
            if hasattr(module, 'mode'):
                module.mode = "quant_forward"
                # module.mode = "quant_weight_only"
