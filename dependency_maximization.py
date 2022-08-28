import torch, os
import numpy as np
from MiniImagenet import MiniImagenet
import scipy.stats
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import random, sys, pickle
import argparse
import sys
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from copy import deepcopy
import math
import torch.nn.init as init
import torch.nn.functional as F
import random
from torch.nn.utils.weight_norm import WeightNorm
import torchvision.models as models
import shutil
from math import cos, pi
import time
import scipy
from numpy.linalg import inv
from tqdm import tqdm


def center(X):
    mean_col = torch.mean(X, dim=0, keepdim=True)
    mean_row = torch.mean(X, dim=1, keepdim=True)
    mean_all = torch.mean(X)
    return X - mean_col - mean_row + mean_all


class LinearKernel(nn.Module):
    def __init__(self):
        super(LinearKernel, self).__init__()
    def forward(self, x):
        return torch.matmul(x, x.t())
    

class GaussianKernel(nn.Module):
    def __init__(self, sigma):
        super(GaussianKernel, self).__init__()
        assert sigma > 0
        self.sigma = sigma
    def forward(self, x):
        X_inner = torch.matmul(x, x.t())
        X_norm = torch.diag(X_inner, diagonal=0)
        X_dist_sq = X_norm + torch.reshape(X_norm, [-1,1]) - 2 * X_inner
        return torch.exp( - X_dist_sq / (2 * self.sigma**2))
    

class HSICLoss(nn.Module):
    def __init__(self, sigma=1, mean_sub=False):
        super(HSICLoss, self).__init__()
        self.kernelX = GaussianKernel(sigma)
        self.kernelY = GaussianKernel(sigma)
        self.mean_sub = mean_sub
    def forward(self, x, y):
        '''
        x: feature
        y: softmax prediction
        '''
        if self.mean_sub is True:
            x = x - torch.mean(x, dim=0) / (torch.std(x, dim=0) + 1e-12)
            y = y - torch.mean(y, dim=0)
        G_X = center(self.kernelX(x))
        G_Y = center(self.kernelY(y))
        res = torch.trace(torch.matmul(G_X, G_Y))
        return res


class CondEntLoss(nn.Module):
    def __init__(self):
        super(CondEntLoss, self).__init__()
    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(1).mean(0)
        return b
    

class EntLoss(nn.Module):
    def __init__(self):
        super(EntLoss, self).__init__()
    def forward(self, x):
        probs = F.softmax(x, dim=1)
        m_probs = probs.mean(0)
        b = m_probs * torch.log(m_probs)
        b = -1.0 * b.sum()
        return b