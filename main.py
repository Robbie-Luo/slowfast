import argparse
import sys
import torch
import os
import slowfast.utils.checkpoint as cu
import slowfast.utils.multiprocessing as mpu
from slowfast.config.defaults import get_cfg
from slowfast.datasets import loader

print("123")
input()
print("456")
cfg = get_cfg()
cfg_file = 'configs/VidOR/C2D_8x8_R50_SHORT.yaml'
cfg.merge_from_file(cfg_file)
train_loader = loader.construct_loader(cfg, "train")
for cur_iter, (inputs, labels, _, meta) in enumerate(train_loader):
    print(inputs)