
import torch
import torch.nn as nn



class BasicNet(nn.Module):

    def __init__(self):
        raise NotImplementedError


    def forward(self, X):
        return torch.rand((7,)).numpy(), torch.rand((1,)).item()