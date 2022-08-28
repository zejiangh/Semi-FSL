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

from model import *
from dependency_maximization import *
from pseudo_label_selection import *
from utils import *


def SSFSL(backbone, mini_test, args):
    acc_all_test = []
    db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=4, pin_memory=True)
    for task_id, (x_spt, y_spt, x_qry, y_qry) in enumerate (db_test):
        start_time = time.time()

        # task model
        model = deepcopy(backbone)

        # task data
        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).cuda(), y_spt.squeeze(0).cuda(), x_qry.squeeze(0).cuda(), y_qry.squeeze(0).cuda()
        x_spt_per, y_spt_per = self_permute(x_spt, y_spt)
        x_spt_per, y_spt_per = x_spt_per.cuda(), y_spt_per.cuda()
        x_qry_per, y_qry_per = self_permute(x_qry, y_qry)
        x_qry_per, y_qry_per = x_qry_per.cuda(), y_qry_per.cuda()

        # separate query and unlabeled set
        classLabels = np.unique(y_qry_per.cpu().numpy())
        x_qry = []
        y_qry = []
        x_ub = []
        y_ub = []
        for label in classLabels:
            classIdx = np.argwhere(y_qry_per.cpu().numpy()==label).T[0]
            xL = x_qry_per[classIdx,:]
            yL = y_qry_per[classIdx]
            x_qry.append(xL[:args.k_qry,:].cpu().numpy())
            y_qry.append(yL[:args.k_qry].cpu().numpy())
            x_ub.append(xL[args.k_qry:,:].cpu().numpy())
            y_ub.append(yL[args.k_qry:].cpu().numpy())
        x_qry = np.asarray(x_qry).reshape((args.k_qry*args.n_way,3,80,80))
        y_qry = np.asarray(y_qry).reshape((args.k_qry*args.n_way,))
        x_ub = np.asarray(x_ub).reshape((args.k_ub*args.n_way,3,80,80))
        y_ub = np.asarray(y_ub).reshape((args.k_ub*args.n_way,))
        x_qry = torch.from_numpy(x_qry).float().cuda()
        y_qry = torch.from_numpy(y_qry).long().cuda()
        x_ub = torch.from_numpy(x_ub).float().cuda()
        y_ub = torch.from_numpy(y_ub).long().cuda()
        if args.mode == 'transductive':
            x_ub = x_qry.clone()
            y_ub = y_qry.clone()

        # feature extracion
        x_spt = x_spt_per.clone()
        y_spt = y_spt_per.clone()
        with torch.no_grad():
            f_spt,_ = model(x_spt)
            f_qry,_ = model(x_qry)
            f_ub,_ = model(x_ub)

        # pseudo-label selection
        f_spt, y_spt = ICI(model, f_spt, y_spt, f_ub, y_ub, f_qry, y_qry, args)
    
        # dependency maximization loss
        xent = nn.CrossEntropyLoss().cuda()
        if args.LS:
            xent = LabelSmoothing(0.1).cuda()
        cent = CondEntLoss().cuda()
        ent = EntLoss().cuda()
        dep = HSICLoss(sigma=args.sigma, mean_sub=True).cuda()

        # classifier
        classifier = CELinear(640, args.n_way).cuda()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=args.update_lr)

        # init classifier
        if args.classifier_init:
            W,b = init_classifier(f_spt, y_spt, args.temperature)
            classifier.L.weight.data = W.clone()
            classifier.L.bias.data = b.clone()

        # classifier training
        classifier.train()
        for step in range (args.update_step):
            o_spt = classifier(f_spt)
            o_qry = classifier(f_qry)
            loss_type = args.loss_type
            assert f_qry.shape[0] == 75
            if loss_type == 'MI':
                loss = 1 * xent(o_spt, y_spt) - args.MI_factor * (ent(o_qry) - 0.1 * cent(o_qry))
            elif loss_type == 'DM':
                loss = 1 * xent(o_spt, y_spt) - args.DM_factor * dep(f_qry, F.softmax(o_qry, dim=1))
            elif loss_type == 'hybrid':
                loss = 1 * xent(o_spt, y_spt) - args.DM_factor * dep(f_qry, F.softmax(o_qry, dim=1)) - args.MI_factor * (ent(o_qry) - 0.1 * cent(o_qry))
            else:
                loss = 1 * xent(o_spt, y_spt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # classifier inference
        classifier.eval()
        output = classifier(f_qry)
        prec1,_ = accuracy(output.data, y_qry, topk=(1,2))
        acc_all_test.append(prec1)
        print('Task:', task_id, 'Acc:', prec1.item(), 'Time:', time.time()-start_time)

    return np.array(acc_all_test).mean(axis=0).astype(np.float32)


def main():

    print(args)

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    
    mini_train = MiniImagenet('/tigress/zejiangh/MiniImageNet/', mode='train', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry, batchsz=args.meta_batch, resize=args.imgsz, data_aug=args.data_aug)
    
    mini_test = MiniImagenet('/tigress/zejiangh/MiniImageNet/', mode='test', n_way=args.test_nway, k_shot=args.k_spt,
                             k_query=args.k_qry + args.k_ub, batchsz=args.test_batch, resize=args.imgsz, data_aug=args.data_aug)

    backbone = Wide_ResNet(num_classes=64).cuda()
    ckpt = torch.load('/home/zejiangh/MAML/semi-supervised/mini/pretrain_wrn28_mini/model_best_tim.pth.tar')
    backbone.load_state_dict(process_state_dict(ckpt['state_dict']))
    backbone.linear = Identity()
    backbone.eval()
    print('Few-shot Testing Accuracy:', SSFSL(backbone, mini_test, args))


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=80)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--data_aug', action='store_true', default=False)
    
    argparser.add_argument('--epochs', type=int, help='iteration number', default=100)
    argparser.add_argument('--pretrain_batchsize', type=int, default=128)
    argparser.add_argument('--lr', type=float, help='pre_training learning rate', default=0.1)
    argparser.add_argument('--lr_decay', type=str, default='cos', help='mode for learning rate decay')
    argparser.add_argument('--update_step', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--beta', type=float, default=0.2)
    argparser.add_argument('--smoothing', type=float, default=0.1)
    argparser.add_argument('--classifier', default='linear', type=str)

    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--test_nway', type=int, default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-4)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=5e-2)
    argparser.add_argument('--inner_step', type=int, default=5)
    argparser.add_argument('--update_step_test', type=int, default=10)
    argparser.add_argument('--test_batch', type=int, default=600)
    argparser.add_argument('--meta_batch', type=int, default=600)
    argparser.add_argument('--task_num', type=int, default=4)
    argparser.add_argument('--temperature', type=int, default=64)
    argparser.add_argument('--type', default='attenuator', type=str)
    
    argparser.add_argument('--k_ub', type=int, help='k shot for query set', default=0)
    argparser.add_argument('--component', type=int, default=10)
    argparser.add_argument('--rho', type=float, default=1e-3)
    argparser.add_argument('--round', type=int, default=1)
    argparser.add_argument('--mode', type=str, default='semi')
    argparser.add_argument('--partition_unlabel', action='store_true', default=False)
    
    argparser.add_argument('--DM_factor', type=float, default=0.01)
    argparser.add_argument('--MI_factor', type=float, default=1)
    argparser.add_argument('--sigma', type=float, default=0.5)
    argparser.add_argument('--loss_type', type=str, default='hybrid', help='MI | DM | hybrid')
    argparser.add_argument('--LS', action='store_true', default=False)
    argparser.add_argument('--classifier_init', action='store_true', default=False)
    
    argparser.add_argument('--seed', type=int, help='random seed', default=1)
    argparser.add_argument('--test_freq', type=int, default=500)
    argparser.add_argument('--print_freq', type=int, default=10)
    argparser.add_argument('--save', default='./logs', type=str, metavar='PATH', help='path to save prune model (default: current directory)')
    
    args = argparser.parse_args()
    main()
