import argparse
import sys
import torch
import os
import slowfast.utils.checkpoint as cu
import numpy as np
# import slowfast.utils.multiprocessing as mpu
from slowfast.config.defaults import get_cfg
from slowfast.datasets import loader
cfg = get_cfg()
# cfg_file = 'configs/VidOR/I3D_8x8_R50.yaml'
cfg_file =  'configs/VidOR/SLOWFAST_32x2_R50_SHORT.yaml'
cfg.merge_from_file(cfg_file)
train_loader = loader.construct_loader(cfg, "train")
val_loader = loader.construct_loader(cfg, "val")
test_loader = loader.construct_loader(cfg, "test")
for cur_iter, (inputs, labels, _, meta) in enumerate(train_loader):
    print(f'Input:{np.shape(inputs[0])}')
    print(f'Label:{np.shape(labels)}')