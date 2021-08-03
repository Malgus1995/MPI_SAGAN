import torch
import torchvision
import torch.nn as nn
import numpy as np



class RefinementNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64,3,1,1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 1,3,1,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)+x
        out = self.sigmoid(out)
        return out
    
    
    
input_test = torch.randn(1,1,106,270)


rfm =  RefinementNet()

res = rfm.forward(input_test)