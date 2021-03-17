import torch 
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .loss_utils import *

def loss_jocor(y1, y2, labels, forget_rate, co_lambda=0.1):
    loss_pick_1 = F.cross_entropy(y1, labels, reduce = False) * (1 - co_lambda)
    loss_pick_2 = F.cross_entropy(y2, labels, reduce = False) * (1 - co_lambda)
    loss_pick = (loss_pick_1 + loss_pick_2 + co_lambda * kl_loss_compute(y1, y2,reduce=False) + co_lambda * kl_loss_compute(y2, y1, reduce=False)).cpu()

    ind_sorted = np.argsort(loss_pick.data)
    loss_sorted = loss_pick[ind_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_sorted))

    ind_update = ind_sorted[:num_remember]

    loss = torch.mean(loss_pick[ind_update])

    return loss, loss