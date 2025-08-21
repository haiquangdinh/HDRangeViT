'''
Adapted from R. Strudel et al.
https://github.com/rstrudel/segmenter

MIT License
Copyright (c) 2021 Robin Strudel
Copyright (c) INRIA
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_

def approximately_clone_state_dict(dst_state_dict, src_state_dict):
    for key in dst_state_dict.keys():
        if key in src_state_dict:
            # Check if the shapes match before copying
            if dst_state_dict[key].shape == src_state_dict[key].shape:
                dst_state_dict[key].copy_(src_state_dict[key])
    return dst_state_dict


def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
