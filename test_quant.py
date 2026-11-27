import time
import warnings
# from imp import new_module
from quant_layers import GPTQConv2D, GPTQLinear, MinMaxQuantConv2d, MinMaxQuantLinear, MinMaxQuantMatMul
import importlib
warnings.filterwarnings("ignore") 
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



def get_args_parser(): 
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
    # parser.add_argument('--optimize', action='store_false', help="Optimize the model") 
   
    calibrate_mode_group = parser.add_mutually_exclusive_group() 
    calibrate_mode_group.add_argument('--calibrate', action='store_true', default=True, help="Calibrate the model")
    calibrate_mode_group.add_argument('--load-calibrate-checkpoint', type=str,
                                      # default='./quant_result/20250909_0844/deit_tiny_w4_a4_s4_calibsize_32.pth', 
                                      default=None,
                                      help="Path to the calibrated checkpoint.")
    parser.add_argument('--test-calibrate-checkpoint', action='store_true', default=False, help='validate the calibrated checkpoint.')

    optimize_mode_group = parser.add_mutually_exclusive_group() 
    optimize_mode_group.add_argument('--optimize', action='store_true', default=False, help="Optimize the model")
    optimize_mode_group.add_argument('--load-optimize-checkpoint', type=str, default=None,
                                     help="Path to the optimized checkpoint.")
    parser.add_argument('--test-optimize-checkpoint', action='store_true', help='validate the optimized checkpoint.')
    return parser


def finish_training(model):
    '''这个主要是针对mlp.fc2这一类Linear层的'''
    
    for name, module in model.named_modules():
        if hasattr(module, 'mode') and hasattr(module, 'reparam_bias'):
            module.reparam_bias()

def save_model(model, args, cfg, mode='calibrate'):
    assert mode in ['calibrate', 'optimize']
    state = {
        "model": model.state_dict(), 
        "hessian": {}, 
        "col":{}
    }
    for name, module in model.named_modules():
        if hasattr(module, "w_quantizer") :
           
            state["hessian"][name] = module.w_quantizer.hessian
            state["col"][name] = module.w_quantizer.cols
        

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
        if isinstance(module, nn.Linear) and 'reduction' in name: 
            module.bias = nn.Parameter(torch.zeros(module.out_features))
        quantizer_attrs = ['a_quantizer', 'w_quantizer', 'A_quantizer', 'B_quantizer']
        for attr in quantizer_attrs:
            if hasattr(module, attr):
                getattr(module, attr).inited = True
    ckpt = torch.load(ckpt_path,map_location=device)
    result = model.load_state_dict(ckpt['model'], strict=False)
    
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
   

    logging.info(f"type(module) = {type(module).__name__}")
    logging.info(f"module.w_quantizer = {type(module.w_quantizer).__name__}") # GPTQQuantizer
    # logging.info(f"module.w_quantizer.scale.shape = {module.w_quantizer.scale.shape}")
    # logging.info(f"module.w_quantizer.zero_point.shape = {module.w_quantizer.zero_point.shape}")
    logging.info(f"module.weight.shape = {module.weight.shape}")
    logging.info(f"module.quantized_weight.shape = {module.quantized_weight.shape}")
    
    torch.save(module.weight, method+("reparam." if flag else "")+target_layer + '.weight.pth')
    torch.save(module.quantized_weight, method+("reparam." if flag else "")+target_layer + '.quantized_weight.pth')
   
    # torch.save(module.w_quantizer.scale, method+("reparam." if flag else "")+target_layer + '.scale.pth')
    # torch.save(module.w_quantizer.zero_point, method+("reparam." if flag else "")+target_layer+'.zero_point.pth')


def collect_parameters(model,calib_loader):
    device = next(model.parameters()).device
    
    for name, module in model.named_modules():
        if hasattr(module, 'mode'):
            module.mode = "quant_forward"
    for name, module in model.named_modules():
        hooks = []  
        '''拿到指定层的量化参数'''
        if name == target_layer:
            hooks.append(module.register_forward_hook(forward_quant_params_hook))
    
    for inp, target in calib_loader:
        inp = inp.to(device)
        _ = model(inp)
    for hook in hooks:
        hook.remove()
    sys.exit()

def main(args):
    logging.info("{} - start the process.".format(get_cur_time())) 
    logging.info(str(args)) 
    dir_path = os.path.dirname(os.path.abspath(args.config)) 
    if dir_path not in sys.path:
        sys.path.append(dir_path) 
    module_name = os.path.splitext(os.path.basename(args.config))[0] 
    imported_module = importlib.import_module(module_name)
    Config = getattr(imported_module, 'Config')  
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
    full_model = copy.deepcopy(model)  
    full_model.to(args.device).eval()
    model.to(args.device).eval()
    logging.info('Building validation dataloader ...')
    g = ViTImageNetLoaderGenerator(args.dataset_root, args.val_batchsize, args.num_workers, kwargs={"model": model})
    val_loader = g.val_loader()
    criterion = nn.CrossEntropyLoss().to(args.device)
    # reparam = args.load_calibrate_checkpoint is None and args.load_optimize_checkpoint is None  
    '''下面是包装整个网络'''
    logging.info('Wraping quantiztion modules ...')
    
    reparam = args.load_calibrate_checkpoint is None and args.load_optimize_checkpoint is None
    model = wrap_modules_in_net(model,cfg,reparam) 
    model.to(args.device).eval()
    # for name, module in model.named_modules():
    #     if hasattr(module, 'mode'):
    #         print(f"name: {name}, mode: {module.mode}")
    #
    # print(model)
    # sys.exit()


    if not args.load_optimize_checkpoint: 
        if args.load_calibrate_checkpoint: 
            logging.info(f"Restoring checkpoint from '{args.load_calibrate_checkpoint}'")
            model = load_model(model, args, args.device, mode='calibrate') 
            # for name, module in model.named_modules():
            #     if isinstance(module, MinMaxQuantLinear) and 'attn.qkv' in name:
            #         print(f"module.w_quantizer.scale.shape = {module.w_quantizer.scale.shape}")
            #         # print(f"name:{name}")
            #         # print(type(name))
            # sys.exit()
            if args.test_calibrate_checkpoint: 
                val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, device=args.device)
        else: 
            logging.info("{} - start calibration".format(get_cur_time()))
            calib_loader = g.calib_loader(num=cfg.calib_size, batch_size=cfg.calib_batch_size, seed=args.seed) 
            quant_calibrator = QuantCalibrator(model, calib_loader)
            '''使用pytorch profile分析量化的时间分布'''

            quant_calibrator.batching_quant_calib() 

            model = wrap_reparamed_modules_in_net(model) #
            model.to(args.device)
            '''这里加一个保存量化参数的方法——加到这里没什么用，因为无法触发。——现在应该能触发了'''
            # collect_parameters(model, calib_loader)
            # sys.exit()
            logging.info("{} - calibration finished.".format(get_cur_time()))
            
            if not args.optimize: 
                finish_training(model) 
            
            logging.info('Validating after calibration ...')
            '''只量化，不推理'''

            val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion,  device=args.device)

    if args.optimize: 
        logging.info('Building calibrator ...')
        calib_loader = g.calib_loader(num=cfg.optim_size, batch_size=cfg.optim_batch_size, seed=args.seed) 
        logging.info("{} - start block reconstruction".format(get_cur_time()))
        block_reconstructor = BlockReconstructor(model, full_model, calib_loader) 
        '''原版代码：block_reconstructor.reconstruct_model(quant_act=cfg.train_act, keep_gpu=cfg.keep_gpu) # block重建'''
        block_reconstructor.reconstruct_model(quant_act=cfg.train_act,keep_gpu=cfg.keep_gpu) 
        finish_training(model) 
        logging.info("{} - block reconstruction finished.".format(get_cur_time()))
        # save_model(model, args, cfg, mode='optimize')
    if args.load_optimize_checkpoint:
        logging.info('Building calibrator ...')
        calib_loader = g.calib_loader(num=cfg.optim_size, batch_size=cfg.optim_batch_size, seed=args.seed) 
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
