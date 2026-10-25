import time
import warnings
# from imp import new_module
from quant_layers import GPTQConv2D, GPTQLinear, MinMaxQuantConv2d, MinMaxQuantLinear, MinMaxQuantMatMul
import importlib
warnings.filterwarnings("ignore") # 禁用所有警告
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
import torch
from utils.models import get_net
from utils.datasets import ViTImageNetLoaderGenerator
from utils.wrap_net import wrap_modules_in_net,wrap_reparamed_modules_in_net
from utils.test_utils import validate
from utils.calibrator import QuantCalibrator

# from quantizers.gptq_quant import GPTQQuantizer
# from quantizers.gptq import
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import logging
import argparse
from tqdm import tqdm
from utils.block_recon import BlockReconstructor
import copy
"""
服务器执行命令：
1. 加载校准后的检查点，验证推理准确率，然后优化模型
python test_quant.py --model deit_tiny --dataset-root /shared-public/data/imagenet2012/imagenet2012 --num-workers 10 --load-calibrate-checkpoint ./quant_result/deit_tiny_w4_a4_s4_calibsize_32.pth --test-calibrate-checkpoint  --val-batchsize 512
2. 从头量化，推理，然后优化模型
python test_quant.py --model deit_tiny --dataset-root /shared-public/data/imagenet2012/imagenet2012 --num-workers 10 --val-batchsize 512 --calibrate --optimize
3. 从头量化，推理，不优化
python test_quant.py --model deit_tiny --dataset-root /shared-public/data/imagenet2012/imagenet2012 --num-workers 10 --val-batchsize 512 --calibrate
"""
# torch.autograd.set_detect_anomaly(True)  # 在开头启用
# torch.set_printoptions(precision=8)  # 显示8位小数

def get_cur_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

while True:
    try:
        timestamp = datetime.now()
        formatted_timestamp = timestamp.strftime("%Y%m%d_%H%M")
        root_path = './quant_result/{}'.format(formatted_timestamp)
        # os.makedirs(root_path) # 创建日志对应的文件夹
        break
    except FileExistsError:
        time.sleep(10)
logging.basicConfig(level=logging.INFO,
                    format='%(message)s',
                    handlers=[
                        # logging.FileHandler('{}/output.log'.format(root_path)), # 将日志写入文件
                        logging.StreamHandler()
                    ])



def get_args_parser(): # 虽然定义了各种参数，但是没用上。现在用上了，测试的时候从这里改参数就行了
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", default='deit_tiny',
                        choices=['vit_small', 'vit_base', 'vit_large',
                                 'deit_tiny', 'deit_small', 'deit_base',
                                  'swin_small', 'swin_base'],
                        help="model")
    # parser.add_argument("--dataset-name", default="imagenet",choices=['coco','imagenet'],help="dataset name")
    parser.add_argument("--dataset-root", default="C:\\Users\\23758\\Desktop\\Dateset\\imagenet",help="dataset root")
    parser.add_argument("--calib-size", default=32, type=int, help="size of calibration set")
    parser.add_argument("--calib-batch-size", default=32, type=int, help="batchsize of calibration set")
    parser.add_argument("--val-batchsize", default=256, type=int, help="batchsize of test set")
    parser.add_argument("--num-workers", default=0, type=int, help="number of data loading workers (default: 0)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else 'cpu', type=str, help="device")
    parser.add_argument("--seed", default=47, type=int, help="seed")
    parser.add_argument('--config', type=str, default="./configs/4bit.py", help="File path to import Config class from")
    # parser.add_argument('--optimize', action='store_false', help="Optimize the model") # 在命令行指定--optimize时store_true表示为true，否则为false
    # 加入是否加载检查点的控制分支
    calibrate_mode_group = parser.add_mutually_exclusive_group()  # 注意，这里的calibrate_mode_group参数之间存在互斥关系
    calibrate_mode_group.add_argument('--calibrate', action='store_true', default=True, help="Calibrate the model")
    calibrate_mode_group.add_argument('--load-calibrate-checkpoint', type=str,
                                      # default='./quant_result/20250909_0844/deit_tiny_w4_a4_s4_calibsize_32.pth', # 这次加载确定日期的检查点。
                                      default=None,
                                      help="Path to the calibrated checkpoint.")
    parser.add_argument('--test-calibrate-checkpoint', action='store_true', default=False, help='validate the calibrated checkpoint.')

    optimize_mode_group = parser.add_mutually_exclusive_group()  # 从头量化和加载检查点互斥
    optimize_mode_group.add_argument('--optimize', action='store_true', default=False, help="Optimize the model")
    optimize_mode_group.add_argument('--load-optimize-checkpoint', type=str, default=None,
                                     help="Path to the optimized checkpoint.")
    parser.add_argument('--test-optimize-checkpoint', action='store_true', help='validate the optimized checkpoint.')
    return parser


def finish_training(model):
    '''这个主要是针对mlp.fc2这一类Linear层的'''
    # logging.info('调用finish_training方法了')
    for name, module in model.named_modules():
        if hasattr(module, 'mode') and hasattr(module, 'reparam_bias'):
            module.reparam_bias()

def save_model(model, args, cfg, mode='calibrate'):
    assert mode in ['calibrate', 'optimize']
    state = {
        "model": model.state_dict(),  # 保存常规参数
        "hessian": {}, # 单独存放hessian矩阵
        "col":{} # 单独存放col值
    }
    for name, module in model.named_modules():
        if hasattr(module, "w_quantizer") :
            # print(f"具有H属性的层：{type(module).__name__}")
            state["hessian"][name] = module.w_quantizer.hessian
            state["col"][name] = module.w_quantizer.cols
        # if hasattr(module,"a_quantizer"): # 有a_quantizer就一定有scale和zero_point
        #     print(f"name: {name}, a_quantier.scale.shape={module.a_quantizer.scale.shape}, a_quantier.zero_point.shape={module.a_quantizer.zero_point.shape}")

    if mode == 'calibrate':
        auto_name = '{}_w{}_a{}_s{}_calibsize_{}.pth'.format(
            args.model, cfg.w_bit, cfg.a_bit, cfg.s_bit, cfg.calib_size)
    else:
        auto_name = '{}_w{}_a{}_s{}_optimsize_{}.pth'.format(
            args.model, cfg.w_bit, cfg.a_bit, cfg.s_bit, cfg.optim_size)
    save_path = os.path.join(root_path, auto_name)

    logging.info(f"Saving checkpoint to {save_path}")
    # torch.save(model.state_dict(), save_path)
    torch.save(state, save_path)


def load_model(model, args, device, mode='calibrate'):
    assert mode in ['calibrate', 'optimize']
    ckpt_path = args.load_calibrate_checkpoint if mode == 'calibrate' else args.load_optimize_checkpoint
    for name, module in model.named_modules():
        if hasattr(module, 'mode'):
            module.calibrated = True
            module.mode = 'quant_forward'
            # module.mode = 'quant_weight_only'
        if isinstance(module, nn.Linear) and 'reduction' in name: # 针对swin的特殊处理
            module.bias = nn.Parameter(torch.zeros(module.out_features))
        quantizer_attrs = ['a_quantizer', 'w_quantizer', 'A_quantizer', 'B_quantizer']
        for attr in quantizer_attrs:
            if hasattr(module, attr):
                getattr(module, attr).inited = True
    ckpt = torch.load(ckpt_path,map_location=device)
    result = model.load_state_dict(ckpt['model'], strict=False)
    # 恢复Hessian矩阵
    if 'hessian' or 'col' in ckpt:
        for name, module in model.named_modules():
            if name in ckpt["hessian"] and name in ckpt["col"]:
                module.w_quantizer.hessian = ckpt["hessian"][name]
                module.w_quantizer.cols = ckpt["col"][name]
    logging.info(str(result))
    model.to(device)
    model.eval()
    return model

# method = "AdaLog."
method = "GPTQ."
# flag = True
flag = False
target_layer = "blocks.0.mlp.fc2"
def forward_quant_params_hook(module, input, output):
    '''这个方法是用来在量化的过程中拿到中间某个层的量化参数'''
    # 1.先看看能不能拿到某一层的数据吧

    logging.info(f"type(module) = {type(module).__name__}") # GPTQAsy...ChannelWise...near，能通过module拿到量化参数
    logging.info(f"module.w_quantizer = {type(module.w_quantizer).__name__}") # GPTQQuantizer
    # logging.info(f"module.w_quantizer.scale.shape = {module.w_quantizer.scale.shape}")
    # logging.info(f"module.w_quantizer.zero_point.shape = {module.w_quantizer.zero_point.shape}")
    logging.info(f"module.weight.shape = {module.weight.shape}")
    logging.info(f"module.quantized_weight.shape = {module.quantized_weight.shape}")
    # 保存一下量化前后的权重
    torch.save(module.weight, method+("reparam." if flag else "")+target_layer + '.weight.pth')
    torch.save(module.quantized_weight, method+("reparam." if flag else "")+target_layer + '.quantized_weight.pth')
    # 保存量化参数
    # torch.save(module.w_quantizer.scale, method+("reparam." if flag else "")+target_layer + '.scale.pth')
    # torch.save(module.w_quantizer.zero_point, method+("reparam." if flag else "")+target_layer+'.zero_point.pth')


def collect_parameters(model,calib_loader):
    device = next(model.parameters()).device
    # 先切换成量化模式
    for name, module in model.named_modules():
        if hasattr(module, 'mode'):
            module.mode = "quant_forward"
    for name, module in model.named_modules():
        hooks = []  # 注册hook，会监听该模块完整的前向计算过程，包括输入和输出
        '''拿到指定层的量化参数'''
        if name == target_layer:
            hooks.append(module.register_forward_hook(forward_quant_params_hook))
    # 前向传播，用于触发hook
    for inp, target in calib_loader:
        inp = inp.to(device)
        _ = model(inp)
    for hook in hooks:
        hook.remove()
    sys.exit()

def main(args):
    logging.info("{} - start the process.".format(get_cur_time()))  # 记录时间
    logging.info(str(args))  # 记录命令行参数
    dir_path = os.path.dirname(os.path.abspath(args.config))  # 获取配置文件所在的目录
    if dir_path not in sys.path:
        sys.path.append(dir_path)  # 加入python路径
    module_name = os.path.splitext(os.path.basename(args.config))[0]  # 这里是提取配置文件的文件名，但是要排除扩展名
    imported_module = importlib.import_module(module_name)
    Config = getattr(imported_module, 'Config')  # 量化的参数配置在Config类中定义，匹配字符串得到参数配置
    logging.info("Successfully imported Config class!")
    cfg = Config()
    cfg.calib_size = args.calib_size if hasattr(args, 'calib_size') else cfg.calib_size
    cfg.calib_batch_size = args.calib_batch_size if hasattr(args, 'calib_batch_size') else cfg.calib_batch_size
    cfg.w_bit = args.w_bit if hasattr(args, 'w_bit') else cfg.w_bit
    cfg.a_bit = args.a_bit if hasattr(args, 'a_bit') else cfg.a_bit
    cfg.s_bit = args.s_bit if hasattr(args, 's_bit') else cfg.s_bit
    for name, value in vars(cfg).items():
        logging.info(f"{name}: {value}")
    # sys.exit()


    model_zoo = {
        'vit_small' : 'vit_small_patch16_224',
        'vit_base'  : 'vit_base_patch16_224',
        'vit_large' : 'vit_large_patch16_224',

        'deit_tiny': 'deit_tiny_patch16_224',
        'deit_small': 'deit_small_patch16_224',
        'deit_base' : 'deit_base_patch16_224',

        'swin_tiny' : 'swin_tiny_patch4_window7_224',
        'swin_small': 'swin_small_patch4_window7_224',
        'swin_base' : 'swin_base_patch4_window7_224',
        'swin_base_384': 'swin_base_patch4_window12_384',
    }
    logging.info('Building model ...')
    name = model_zoo.get(args.model)
    model = get_net(name)
    full_model = copy.deepcopy(model)  # 创建未量化模型的完整模型副本，为什么要这样？在重建过程中是需要全精度模型的，我用不到。
    full_model.to(args.device).eval()
    model.to(args.device).eval()
    logging.info('Building validation dataloader ...')
    g = ViTImageNetLoaderGenerator(args.dataset_root, args.val_batchsize, args.num_workers, kwargs={"model": model})
    val_loader = g.val_loader()
    criterion = nn.CrossEntropyLoss().to(args.device)
    # reparam = args.load_calibrate_checkpoint is None and args.load_optimize_checkpoint is None  # 判断是否需要重新参数化（True表示从头开始量化，False表示加载已有量化检查点）
    '''下面是包装整个网络'''
    logging.info('Wraping quantiztion modules ...')
    # 这个reparam参数作用是在加载校准检查点时，控制模型包装不使用重参数化
    reparam = args.load_calibrate_checkpoint is None and args.load_optimize_checkpoint is None
    model = wrap_modules_in_net(model,cfg,reparam) # 这里包装的模型还是通道级的量化参数,但是检查点则是层级的量化参数
    model.to(args.device).eval()
    # for name, module in model.named_modules():
    #     if hasattr(module, 'mode'):
    #         print(f"name: {name}, mode: {module.mode}")
    #
    # print(model)
    # sys.exit()


    if not args.load_optimize_checkpoint: # 如果未加载优化后的检查点
        if args.load_calibrate_checkpoint: # 如果加载已校准的检查点。注意区分优化和校准
            logging.info(f"Restoring checkpoint from '{args.load_calibrate_checkpoint}'")
            model = load_model(model, args, args.device, mode='calibrate') # 加载校准后的模型权重
            # for name, module in model.named_modules():
            #     if isinstance(module, MinMaxQuantLinear) and 'attn.qkv' in name:
            #         print(f"module.w_quantizer.scale.shape = {module.w_quantizer.scale.shape}")
            #         # print(f"name:{name}")
            #         # print(type(name))
            # sys.exit()
            if args.test_calibrate_checkpoint: # 如果要求测试校准模型
                val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, device=args.device)
        else: # 如果未加载校准的检查点，需要校准
            logging.info("{} - start calibration".format(get_cur_time()))
            calib_loader = g.calib_loader(num=cfg.calib_size, batch_size=cfg.calib_batch_size, seed=args.seed) # 校准集
            quant_calibrator = QuantCalibrator(model, calib_loader) # 量化校准器
            '''使用pytorch profile分析量化的时间分布'''

            quant_calibrator.batching_quant_calib() # 批量校准

            model = wrap_reparamed_modules_in_net(model) # 对已经校准（统计过数据分布）的量化模型进行参数固化，将临时统计值转换为可部署的量化参数
            model.to(args.device)
            '''这里加一个保存量化参数的方法——加到这里没什么用，因为无法触发。——现在应该能触发了'''
            # collect_parameters(model, calib_loader)
            # sys.exit()
            logging.info("{} - calibration finished.".format(get_cur_time()))
            # 当用户仅执行校准（--calibrate）而不进行后续优化（--optimize）时，立即固化模型的所有量化参数。模型从"可训练状态"转为"仅推理状态"，量化参数不再更新。
            if not args.optimize: # 没有启用优化流程
                finish_training(model) # 冻结模型参数
            # save_model(model, args, cfg, mode='calibrate')
            logging.info('Validating after calibration ...')
            '''只量化，不推理'''

            val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion,  device=args.device)

    if args.optimize: # 启用优化流程
        logging.info('Building calibrator ...')
        calib_loader = g.calib_loader(num=cfg.optim_size, batch_size=cfg.optim_batch_size, seed=args.seed) # 准备数据加载器
        logging.info("{} - start block reconstruction".format(get_cur_time()))
        block_reconstructor = BlockReconstructor(model, full_model, calib_loader) # 实例化一个block级的重建器
        '''原版代码：block_reconstructor.reconstruct_model(quant_act=cfg.train_act, keep_gpu=cfg.keep_gpu) # block重建'''
        block_reconstructor.reconstruct_model(quant_act=cfg.train_act,keep_gpu=cfg.keep_gpu) # 我的代码
        finish_training(model) # 冻结模型参数
        logging.info("{} - block reconstruction finished.".format(get_cur_time()))
        # save_model(model, args, cfg, mode='optimize') # 保存模型
    if args.load_optimize_checkpoint: # 加载优化后的模型
        logging.info('Building calibrator ...')
        calib_loader = g.calib_loader(num=cfg.optim_size, batch_size=cfg.optim_batch_size, seed=args.seed) # 又来一个数据加载器
        model = load_model(model, args, args.device, mode='optimize')
    if args.optimize or args.test_optimize_checkpoint:
        logging.info('Validating on calibration set after block reconstruction ...')
        val_loss, val_prec1, val_prec5 = validate(calib_loader, model, criterion, device=args.device)
        logging.info('Validating on test set after block reconstruction ...')
        val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, device=args.device)
    logging.info("{} - finished the process.".format(get_cur_time()))






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='参数配置')
    args = get_args_parser().parse_args()
    # print(args)
    st_time = time.time()
    main(args)
    # time.sleep(3)
    ed_time = time.time()
    logging.info(f"Total execution time: {str(timedelta(seconds=int(ed_time-st_time)))}")
