import torch 
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .loss_utils import *

def loss_structrue(feat1, feat2):
    q1 = CalPt_norm(Distance_squared(feat1, feat1))
    q2 = CalPt_norm(Distance_squared(feat2, feat2))
    return -1 * (q1 * torch.log(q2 + eps)).mean()