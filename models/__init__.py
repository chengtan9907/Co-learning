from .resnet import resnet18, resnet34, resnet50
# from .preresnet import preresnet18, preresnet34, preresnet50
from .model import Model_r18, Model_r34

__all__ = ('resnet18', 'resnet34', 'resnet50', 
        #    'preresnet18', 'preresnet34', 'preresnet50',
           'Model_r18', 'Model_r34')