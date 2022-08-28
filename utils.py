import logging
import torch, os
import numpy as np
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
from torch.nn.utils.weight_norm import WeightNorm
import torchvision.models as models
import shutil
from math import cos, pi
import time
import scipy
from numpy.linalg import inv
from tqdm import tqdm


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class Grayscale(object):

    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Contrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class ColorJitter(object):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, img):
        self.transforms = []
        if self.brightness != 0:
            self.transforms.append(Brightness(self.brightness))
        if self.contrast != 0:
            self.transforms.append(Contrast(self.contrast))
        if self.saturation != 0:
            self.transforms.append(Saturation(self.saturation))

        random.shuffle(self.transforms)
        transform = Compose(self.transforms)
        # print(transform)
        return transform(img)
    

def process_state_dict(state_dict):
    # process state dict so that it can be loaded by normal models
    for k in list(state_dict.keys()):
        state_dict[k.replace('module.', '')] = state_dict.pop(k)
    return state_dict


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
 
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return pm


class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def self_permute(x_spt, y_spt):
    '''
    x_spt: N X d
    y_spt: N,
    '''
    x_spt = x_spt.detach().cpu().clone().numpy()
    y_spt = y_spt.detach().cpu().clone().numpy()
    x_out = []
    y_out = []
    idx = np.argsort(y_spt)
    for j in range (x_spt.shape[0]):
        x_out.append(x_spt[idx[j],:])
        y_out.append(y_spt[idx[j]])
    x_out = np.asarray(x_out).reshape(x_spt.shape)
    y_out = np.asarray(y_out).reshape(y_spt.shape)
    x_out = torch.from_numpy(x_out).float()
    y_out = torch.from_numpy(y_out).long()
    return x_out, y_out


def init_classifier (f_spt, y_spt, temperature):
    num_classes = torch.unique(y_spt).size(0)
    eye = torch.eye(num_classes).to(y_spt.device)
    one_hot = []
    for y_task in y_spt:
        one_hot.append(eye[y_task].unsqueeze(0))
    one_hot = torch.cat(one_hot, 0)
    counts = one_hot.sum(0).view(-1, 1)
    proto = one_hot.transpose(0, 1).matmul(f_spt)
    proto = proto / counts
    W = 2 * proto / temperature
    b = - torch.norm(proto, dim=1) ** 2 / temperature
    return W, b
    

def get_mi(probs):
    q_cond_ent = get_cond_entropy(probs)
    q_ent = get_entropy(probs)
    return q_ent - q_cond_ent


def get_entropy(probs):
    q_ent = - (probs.mean(1) * torch.log(probs.mean(1) + 1e-12)).sum(1, keepdim=True)
    return q_ent


def get_cond_entropy(probs):
    q_cond_ent = - (probs * torch.log(probs + 1e-12)).sum(2).mean(1, keepdim=True)
    return q_cond_ent

def get_one_hot(y_s):
    num_classes = torch.unique(y_s).size(0)
    eye = torch.eye(num_classes).to(y_s.device)
    one_hot = []
    for y_task in y_s:
        one_hot.append(eye[y_task].unsqueeze(0))
    one_hot = torch.cat(one_hot, 0)
    return one_hot



