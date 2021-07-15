import torch 
from .loss_utils import *

def loss_structrue(feat1, feat2):
    q1 = CalPairwise(Distance_squared(feat1, feat1))
    q2 = CalPairwise(Distance_squared(feat2, feat2))
    return -1 * (q1 * torch.log(q2 + eps)).mean()

def loss_structrue_t(feat1, feat2, v):
    q1 = CalPairwise_t(Distance_squared(feat1, feat1), v)
    q2 = CalPairwise_t(Distance_squared(feat2, feat2), v)
    return -1 * (q1 * torch.log(q2 + eps)).mean()