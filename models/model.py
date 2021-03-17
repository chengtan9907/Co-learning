import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet18
from collections import OrderedDict
from torchvision.models import resnet34, resnet50

class Model_r18(nn.Module):
    def __init__(self, feature_dim=128, is_linear=False, num_classes=None):
        super(Model_r18, self).__init__()
        self.f = OrderedDict([])
        for name, module in resnet18().named_children(): 
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.update({name: module})
                
        # encoder
        self.f = nn.Sequential(self.f)
        # projection head
        self.g = nn.Sequential(
                    nn.Linear(512, 512, bias=False), 
                    nn.ReLU(inplace=True), 
                    nn.Linear(512, feature_dim, bias=True)
                )
        
        self.is_linear = is_linear
        if is_linear == True:
            self.fc = nn.Linear(512, num_classes)

    def forward(self, x, forward_fc=True, ignore_feat=False):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        projection = self.g(feature)
        if self.is_linear and forward_fc:
            logits = self.fc(feature)
            if ignore_feat == True:
                return projection, logits
            else:
                return feature, projection, logits
        else:
            if ignore_feat == True:
                return projection
            else:
                return feature, projection


class Model_r34(nn.Module):
    def __init__(self, feature_dim=128, is_linear=False, num_classes=None):
        super(Model_r34, self).__init__()
        self.f = OrderedDict([])
        for name, module in resnet34().named_children(): 
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.update({name: module})
                
        # encoder
        self.f = nn.Sequential(self.f)
        # projection head
        self.g = nn.Sequential(
                    nn.Linear(512, 512), 
                    nn.ReLU(inplace=True), 
                    nn.Linear(512, feature_dim)
                )
        
        self.is_linear = is_linear
        if is_linear == True:
            self.fc = nn.Linear(512, num_classes)

    def forward(self, x, forward_fc=True, ignore_feat=False):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        projection = self.g(feature)
        if self.is_linear and forward_fc:
            logits = self.fc(feature)
            if ignore_feat == True:
                return projection, logits
            else:
                return feature, projection, logits
        else:
            if ignore_feat == True:
                return projection
            else:
                return feature, projection