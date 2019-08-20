
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

def smoid(x):
    return 1/(1+np.exp(-x))

def to_np(x):
    return x.detach().cpu().numpy()

def avg(x):
    return sum(x)/len(x)

def binarify(target):
    keys = sorted(list(set(target)));
    target = np.float32([[1.0 if entry == key else 0.0
        for key in keys] for entry in target ])
    return target, keys

