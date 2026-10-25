import time
import torch
from torch import nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import logging


# def validate(val_loader, model, criterion, device='cuda:0'):
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()
#
#     # 切换到评估模式
#     model.eval()
#     # tqdm包装验证数据加载器
#     with tqdm(total=len(val_loader), desc="Validating", unit="batch") as pbar:
#         for i, (data, target) in enumerate(val_loader):
#             data = data.to(device)
#             target = target.to(device)
#             with torch.no_grad():
#                 output = model(data)
#             loss = criterion(output, target)  # 计算损失
#
#             # 计算准确率并记录损失
#             prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
#             losses.update(loss.data.item(), data.size(0))
#             top1.update(prec1.data.item(), data.size(0))
#             top5.update(prec5.data.item(), data.size(0))
#
#             # 更新进度条
#             pbar.set_postfix({
#                 "Loss": f"{losses.val:.4f} (AVG:{losses.avg:.4f})",
#                 "Top1": f"{top1.val:.3f} (AVG:{top1.avg:.3f})",
#                 "Top5": f"{top5.val:.3f} (AVG:{top5.avg:.3f})",
#                 # "Time": f"{batch_time.val:.3f}s"
#             })
#             pbar.update(1)
#
#             # 输出日志
#             logging.info('Test: [{0}/{1}]\t'
#                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
#                 i,
#                 len(val_loader),
#                 loss=losses,
#                 top1=top1,
#                 top5=top5,
#             ))
#
#     logging.info(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {losses.avg:.3f}'.
#                  format(top1=top1, top5=top5, losses=losses))
#
#     return losses.avg, top1.avg, top5.avg
import time
import logging

def validate(val_loader, model, criterion, device='cuda:0'):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    start_time = time.time()

    for i, (data, target) in enumerate(val_loader):
        data = data.to(device)
        target = target.to(device)

        with torch.no_grad():
            output = model(data)
            loss = criterion(output, target)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        batch_size = data.size(0)
        losses.update(loss.item(), batch_size)
        top1.update(prec1.item(), batch_size)
        top5.update(prec5.item(), batch_size)

        # 每个批次输出一条信息
        logging.info(
            "Validate: [{}/{}]  "
            "Loss {loss.val:.4f} (AVG:{loss.avg:.4f})  "
            "Prec@1 {top1.val:.3f} (AVG:{top1.avg:.3f})  "
            "Prec@5 {top5.val:.3f} (AVG:{top5.avg:.3f})".format(
                i + 1,
                len(val_loader),
                loss=losses,
                top1=top1,
                top5=top5,
            )
        )

    total_time = time.time() - start_time
    # 最终平均日志
    logging.info(
        "Validation Summary: Total batches: {}  "
        "Elapsed: {:.2f}s  "
        "Avg Loss: {losses.avg:.4f}  "
        "Avg Prec@1: {top1.avg:.3f}  "
        "Avg Prec@5: {top5.avg:.3f}".format(
            len(val_loader),
            total_time,
            losses=losses,
            top1=top1,
            top5=top5,
        )
    )
    return losses.avg, top1.avg, top5.avg





class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
