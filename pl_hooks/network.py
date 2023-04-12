# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/03_network.ipynb.

__all__ = ['conv', 'cnn_layers']

import torch

def conv(ni, nf, ks=3, act=True):
    res = torch.nn.Conv2d(ni, nf, stride=2, kernel_size=ks, padding=ks//2)
    if act: res = torch.nn.Sequential(res, torch.nn.ReLU())
    return res

def cnn_layers():
    return torch.nn.Sequential(
        conv(1 ,8, ks=5),        #14x14
        conv(8 ,16),             #7x7
        conv(16,32),             #4x4
        conv(32,64),             #2x2
        conv(64,10, act=False),  #1x1
        torch.nn.Flatten())