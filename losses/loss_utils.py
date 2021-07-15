import scipy
import torch 
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from scipy import special

eps = 1e-12

def kl_loss_compute(pred, soft_targets, reduce=True):
    kl = F.kl_div(F.log_softmax(pred, dim=1),F.softmax(soft_targets, dim=1),reduce=False)

    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)

def Distance_squared(x, y, featdim=1):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    d = dist.clamp(min=eps)
    d[torch.eye(d.shape[0]) == 1] = eps
    return d

def CalPairwise(dist):
    dist[dist < 0] = 0
    Pij = torch.exp(-dist)
    return Pij

def CalPairwise_t(dist, v):
    C = scipy.special.gamma((v + 1) / 2) / (np.sqrt(v * np.pi) * scipy.special.gamma(v / 2))
    return torch.pow((1 + torch.pow(dist, 2) / v), - (v + 1) / 2)

def CE(P, Q):
    loss = -1 * (P * torch.log(Q + eps)).mean()
    return loss
