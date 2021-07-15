from .loss_jocor import loss_jocor
from .loss_structrue import loss_structrue, loss_structrue_t
from .loss_coteaching import loss_coteaching
from .loss_ntxent import NTXentLoss
from .loss_other import SCELoss, GCELoss, DMILoss

__all__ = ('loss_jocor', 'loss_structrue', 'loss_coteaching', 'NTXentLoss',
           'SCELoss', 'GCELoss', 'DMILoss')