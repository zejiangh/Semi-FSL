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

from dependency_maximization import *
from model import *
from utils import *


def PCA_eig(X, k, center=True, scale=False):
    n,p = X.size()
    ones = torch.ones(n).view([n,1])
    h = ((1/n) * torch.mm(ones, ones.t())) if center else torch.zeros(n*n).view([n,n])
    H = torch.eye(n) - h
    X_center =  torch.mm(H.double(), X.double())
    covariance = 1/(n-1) * torch.mm(X_center.t(), X_center).view(p,p)
    scaling =  torch.sqrt(1/torch.diag(covariance)).double() if scale else torch.ones(p).double()
    scaled_covariance = torch.mm(torch.diag(scaling).view(p,p), covariance)
    eigenvalues, eigenvectors = torch.eig(scaled_covariance, True)
    components = (eigenvectors[:, :k]).t()
    explained_variance = eigenvalues[:k, 0]
    return { 'X':X, 'k':k, 'components':components, 'explained_variance':explained_variance }


def DI (X,y, rho):
    Sb = np.zeros((X.shape[1], X.shape[1]))
    S = np.inner(X.T, X.T)
    N = len(X)
    mu = np.mean(X, axis=0)
    classLabels = np.unique(y)
    for label in classLabels:
        classIdx = np.argwhere(y==label).T[0]
        Nl = len(classIdx)
        xL = X[classIdx]
        muL = np.mean(xL, axis=0)
        muLbar = muL-mu
        Sb = Sb + Nl*np.outer(muLbar,muLbar)
    Sbar = S - N*np.outer(mu,mu)
    Sbar = Sbar + rho * np.eye(np.shape(Sbar)[0])
    return np.trace(np.dot(inv(Sbar), Sb))/(len(classLabels)-1)


def IDA (X,y, n_component, rho):
    PCA_output = PCA_eig(torch.from_numpy(X).float(), n_component)
    X = np.matmul(X, PCA_output['components'].cpu().numpy().T)
    diff = []
    total_DI = DI(X, y, rho)
    for i in range(X.shape[0]):
        X_ = np.delete(X, i, axis=0)
        y_ = np.delete(y, i)
        diff.append(total_DI-DI(X_,y_, rho))
    return np.asarray(diff), total_DI


def classifier_fit(f_spt, y_spt, f_ub, args):
    #loss
    xent = nn.CrossEntropyLoss().cuda()
    if args.LS:
        xent = LabelSmoothing(0.1).cuda()
    cent = CondEntLoss().cuda()
    ent = EntLoss().cuda()
    dep = HSICLoss(sigma=args.sigma, mean_sub=True).cuda()
    # classifier
    classifier = CELinear(640, args.n_way).cuda()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.update_lr)
    # initialize classifier
    if args.classifier_init:
        W,b = init_classifier(f_spt, y_spt, args.temperature)
        classifier.L.weight.data = W.clone()
        classifier.L.bias.data = b.clone()
    # classifier training
    classifier.train()
    for step in range (args.update_step):
        o_spt = classifier(f_spt)
        o_ub = classifier(f_ub)
        loss_type = args.loss_type
        if f_ub.shape[0] > 75:
            loss_type = 'MI'
        if loss_type == 'MI':
            loss = 1 * xent(o_spt, y_spt) - args.MI_factor * (ent(o_ub) - 0.1 * cent(o_ub))
        elif loss_type == 'DM':
            loss = 1 * xent(o_spt, y_spt) - args.DM_factor * dep(f_ub, F.softmax(o_ub, dim=1))
        elif loss_type == 'hybrid':
            loss = 1 * xent(o_spt, y_spt) - args.DM_factor * dep(f_ub, F.softmax(o_ub, dim=1)) - args.MI_factor * (ent(o_ub) - 0.1 * cent(o_ub))
        else:
            loss = 1 * xent(o_spt, y_spt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    classifier.eval()
    return classifier(f_ub)


def ICI (model, f_spt, y_spt, f_ub, y_ub, f_qry, y_qry, args):

    # num_select = []
    # num_flip = []
    # accuracy_on_select = []
    # accuracy_on_all = []
    # accuracy_on_query = []
    # DI_on_all = []
    # prev_pred = y_ub.clone()

    f_spt_temp = f_spt.clone()
    y_spt_temp = y_spt.clone()
    model.eval()
    rounds = args.round
    expand_idx = []
    for i in range(rounds):
        
        prev_length = len(expand_idx)
        
        if len(expand_idx) >= args.n_way*args.k_ub:
            break
        
        outputs = classifier_fit(f_spt, y_spt, f_ub, args)
        y_pred = F.softmax(outputs, dim=1).argmax(dim=1).cuda()
        
        score, total_DI = IDA(f_ub.cpu().numpy(), y_pred.cpu().numpy(), args.component, args.rho)
        rank = np.argsort(score)[::-1].tolist()
        for c in rank:
            if len(expand_idx) >= args.n_way*5*(i+1) or len(expand_idx)>=args.n_way*args.k_ub:
                break
            label = y_pred[c]
            count = 0
            for k in expand_idx:
                if y_pred[k] == label:
                    count += 1
            if count < 5*(i+1) and count < args.k_ub and c not in expand_idx:
                expand_idx.append(c)
                
        print('round', i, 'correct select', torch.sum(y_pred[expand_idx]==y_ub[expand_idx]).item(), 'total select', len(expand_idx), 
              'selected accuracy', torch.sum(y_pred[expand_idx]==y_ub[expand_idx]).item()/len(expand_idx), 'accuracy', torch.sum(y_pred==y_ub).item()/y_pred.shape[0])

        # num_select.append(len(expand_idx))
        # num_flip.append(torch.sum(y_pred != prev_pred).item()/y_pred.shape[0])
        # accuracy_on_select.append(torch.sum(y_pred[expand_idx]==y_ub[expand_idx]).item()/len(expand_idx))
        # accuracy_on_all.append(torch.sum(y_pred==y_ub).item()/y_pred.shape[0])
        # DI_on_all.append(total_DI)
        # prev_pred = y_pred.clone()
        
        f_spt = torch.cat((f_spt_temp, f_ub[expand_idx,:]), dim=0)
        y_spt = torch.cat((y_spt_temp, y_pred[expand_idx]), dim=0)
        
        f_spt, y_spt = self_permute(f_spt, y_spt)
        f_spt, y_spt = f_spt.cuda(), y_spt.cuda()

        # woc = classifier_fit(f_spt, y_spt, f_qry)
        # woc = F.softmax(woc, dim=1).argmax(dim=1).cuda()
        # accuracy_on_query.append(torch.sum(woc==y_qry).item()/woc.shape[0])
        
        if len(expand_idx) == prev_length:
            break

    # task_logger = {}
    # task_logger['num_select'] = num_select
    # task_logger['num_flip'] = num_flip
    # task_logger['accuracy_on_select'] = accuracy_on_select
    # task_logger['accuracy_on_all'] = accuracy_on_all
    # task_logger['accuracy_on_query'] = accuracy_on_query
    # task_logger['DI_on_all'] = DI_on_all
        
    return f_spt, y_spt