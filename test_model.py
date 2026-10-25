import torch
import timm

model = timm.create_model('deit_tiny_patch16_224', pretrained=True)
print(model)