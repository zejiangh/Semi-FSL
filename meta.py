import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import numpy as np
from learner import ClassifierLearner, CosineLearner, ResNet12Learner, MultiHeadAttention, Attenuator
from copy import deepcopy
import sys
from torch.autograd import Variable

def estimate_cov(examples, rowvar=False, inplace=False):
    if examples.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if examples.dim() < 2:
        examples = examples.view(1, -1)
    if not rowvar and examples.size(0) != 1:
        examples = examples.t()
    factor = 1.0 / (examples.size(1) - 1)
    if inplace:
        examples -= torch.mean(examples, dim=1, keepdim=True)
    else:
        examples = examples - torch.mean(examples, dim=1, keepdim=True)
    examples_t = examples.t()
    return factor * examples.matmul(examples_t).squeeze()

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
class BasicBlock3(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock3, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)
        return out
class ResNet12(nn.Module):
    def __init__(self, temperature, shot, way):
        self.inplanes = 3
        super(ResNet12, self).__init__()
        self.temperature = temperature
        self.shot = shot
        self.way = way
        block = BasicBlock3
        self.layer1 = self._make_layer(block, 64, stride=2)
        self.layer2 = self._make_layer(block, 160, stride=2)
        self.layer3 = self._make_layer(block, 320, stride=2)
        self.layer4 = self._make_layer(block, 640, stride=2)
        self.avgpool = nn.AvgPool2d(5, stride=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
    def _make_layer(self, block, planes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        proto = torch.mean(x[0:(self.shot+5), :], dim=0).unsqueeze(0)
        for c_K in range (self.way-1):
            proto = torch.cat((proto, torch.mean(x[(c_K+1)*(self.shot+5):(c_K+2)*(self.shot+5), :], dim=0).unsqueeze(0)), dim=0)
        W = 2 * proto / self.temperature
        b = - torch.norm(proto, dim=1) ** 2 / self.temperature
        x = F.linear(x, W, b)
        return x
    
def conv_score (layer, model, x_spt, y_spt):
    res = []
    def for_hook(module, inputs, outputs):
        res.append(outputs.detach().cpu().numpy())
    handle = layer.register_forward_hook(for_hook)
    pred = model(x_spt)
    handle.remove()
    loss = F.cross_entropy(pred, y_spt)
    loss.backward()
    return layer.weight.grad.detach().cpu().numpy().mean(), res[0].mean()

def bn_score (layer, model, x_spt, y_spt):
    res = []
    def for_hook(module, inputs, outputs):
        res.append(outputs.detach().cpu().numpy())
    handle = layer.register_forward_hook(for_hook)
    pred = model(x_spt)
    handle.remove()
    loss = F.cross_entropy(pred, y_spt)
    loss.backward()
    return layer.weight.grad.detach().cpu().numpy().mean(), layer.bias.grad.detach().cpu().numpy().mean(), res[0].mean(), res[0].mean()
    
def get_mask (model, x_spt, y_spt):
    pre_grad = []
    pre_proto = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            a,b = conv_score(m, model, x_spt, y_spt)
            pre_grad.append(a)
            pre_proto.append(b)
        if isinstance(m, nn.BatchNorm2d):
            a,b,c,d = bn_score(m, model, x_spt, y_spt)
            pre_grad.append(a)
            pre_grad.append(b)
            pre_proto.append(c)
            pre_proto.append(d)
    return pre_grad, pre_proto
    
def zero_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()

class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, model):
        super(Meta, self).__init__()
        
        self.update_lr = args.update_lr
        self.inner_step = args.inner_step
        self.update_step_test = args.update_step_test
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.test_nway = args.test_nway
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.temperature = args.temperature
        self.net = ResNet12Learner(self.n_way, model)
        self.SA = MultiHeadAttention(1, 640, 640, 640, dropout=0.5).cuda()
        self.attenuator = Attenuator(48).cuda()
        self.meta_optim = optim.Adam([{'params': self.net.parameters()}, 
                                      {'params': self.SA.parameters(), 'lr': 10*self.meta_lr},
                                      {'params': self.attenuator.parameters(), 'lr': 10*self.meta_lr}], lr=self.meta_lr)

    def clip_grad_by_norm_(self, grad, max_norm):
        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)
        return total_norm/counter


    def forward(self, x_spt, y_spt, x_qry, y_qry, step, total_step):
        """
        x_spt:   [b, setsz, c_]
        y_spt:   [b, setsz]
        x_qry:   [b, querysz, c_]
        y_qry:   [b, querysz]
        step: for decaying the outer-loop Adam learning rate
        """
        
#        if step in [int(total_step*1/3), int(total_step*2/3)]:
#            for param_group in self.meta_optim.param_groups:
#                param_group['lr'] *= 0.5
#            for param_group in self.meta_optim.param_groups:
#                print('change learning rate to', param_group['lr'])
        
        querysz = x_qry[0].size(0)
        corrects = [0 for _ in range(self.inner_step + 1)]
        adapted_base = []
        adapted_W = []
        adapted_b = []

        for i in range(self.task_num):
            '''
            Generate task-specific attentuator factors
            '''
            fetcher = ResNet12(self.temperature, self.k_spt, self.n_way).cuda()
            idx = 0
            bn_idx = 0
            for m in fetcher.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.data = self.net.vars[idx].clone()
                    idx += 1
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data = self.net.vars[idx].clone()
                    m.bias.data = self.net.vars[idx+1].clone()
                    idx += 2
                    m.running_mean.data = self.net.vars_bn[bn_idx].clone()
                    m.running_var.data = self.net.vars_bn[bn_idx+1].clone()
                    bn_idx += 2
            assert idx == 48
            assert bn_idx == 32
            pre_grad, pre_proto = get_mask(fetcher, x_spt[i], y_spt[i])
            # generate attenuator inputs
            pre_grad, pre_proto = Variable(torch.from_numpy(np.asarray(pre_grad)).float().cuda()), Variable(torch.from_numpy(np.asarray(pre_proto)).float().cuda())
            # output attenuator factors
            gamma = self.attenuator(pre_grad, pre_proto)
            # obtain initial weights
            backbone = list(map(lambda p: p[0] * p[1], zip(gamma, self.net.parameters())))

            """
            Initialize the classifier weight by self-attention
            Assumption: 
            (1) x_spt, y_spt permuted
            (2) 1-shot: each support sample is proto
            """
            feat,_ = self.net(x_spt[i], vars=backbone, fc_weight=None, fc_bias=None, bn_training=True)
            proto = torch.mean(feat[0:(self.k_spt+5), :], dim=0).unsqueeze(0)
            for c_K in range (self.n_way-1):
                proto = torch.cat((proto, torch.mean(feat[(c_K+1)*(self.k_spt+5):(c_K+2)*(self.k_spt+5), :], dim=0).unsqueeze(0)), dim=0)
            slf_attn = self.SA(proto.unsqueeze(0), proto.unsqueeze(0), proto.unsqueeze(0)).squeeze(0)
            W = 2 * slf_attn / self.temperature
            b = - torch.norm(slf_attn, dim=1) ** 2 / self.temperature
            _,logits = self.net(x_spt[i], vars=backbone, fc_weight=W, fc_bias=b, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i])
            zero_grad(backbone)
            zero_grad(W)
            zero_grad(b)
            grad = torch.autograd.grad(loss, backbone, retain_graph=True)
            grad_W_b = torch.autograd.grad(loss, (W,b))
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, backbone)))
            fast_W = W - self.update_lr * grad_W_b[0]
            fast_b = b - self.update_lr * grad_W_b[1]

            with torch.no_grad():
                _,logits_q = self.net(x_qry[i], backbone, W, b, bn_training=True)
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            with torch.no_grad():
                _,logits_q = self.net(x_qry[i], fast_weights, fast_W, fast_b, bn_training=True)
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.inner_step):
                _,logits = self.net(x_spt[i], fast_weights, fast_W, fast_b, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                zero_grad(fast_weights)
                zero_grad(fast_W)
                zero_grad(fast_b)
                grad = torch.autograd.grad(loss, fast_weights, retain_graph=True)
                grad_W_b = torch.autograd.grad(loss, (fast_W, fast_b))
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                fast_W = fast_W - self.update_lr * grad_W_b[0]
                fast_b = fast_b - self.update_lr * grad_W_b[1]
                
                with torch.no_grad():
                    _,logits_q = self.net(x_qry[i], fast_weights, fast_W, fast_b, bn_training=True)
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[k + 1] = corrects[k + 1] + correct
              
            adapted_base.append(fast_weights)
            adapted_W.append(fast_W)
            adapted_b.append(fast_b)

        meta_loss = 0.
        for i in range(self.task_num):
            _,logits = self.net(x_qry[i], adapted_base[i], adapted_W[i], adapted_b[i], bn_training=True)
            meta_loss += F.cross_entropy(logits, y_qry[i])
        meta_loss /= float(self.task_num)
        self.meta_optim.zero_grad()
        meta_loss.backward()
        self.meta_optim.step()
        accs = np.array(corrects) / (querysz * self.task_num)
        return accs
    
    def finetune(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4
        querysz = x_qry.size(0)
        corrects = [0 for _ in range(self.update_step_test + 1)]
        if self.n_way == self.test_nway:
            net = deepcopy(self.net)
        else:
            print('training way not equal to testing way')
            sys.exit()

        fetcher = ResNet12(self.temperature, self.k_spt, self.n_way).cuda()
        idx = 0
        bn_idx = 0
        for m in fetcher.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data = net.vars[idx].clone()
                idx += 1
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data = net.vars[idx].clone()
                m.bias.data = net.vars[idx+1].clone()
                idx += 2
                m.running_mean.data = net.vars_bn[bn_idx].clone()
                m.running_var.data = net.vars_bn[bn_idx+1].clone()
                bn_idx += 2
        assert idx == 48
        assert bn_idx == 32
        pre_grad, pre_proto = get_mask(fetcher, x_spt, y_spt)
        pre_grad, pre_proto = Variable(torch.from_numpy(np.asarray(pre_grad)).float().cuda()), Variable(torch.from_numpy(np.asarray(pre_proto)).float().cuda())
        gamma = self.attenuator(pre_grad, pre_proto)
        backbone = list(map(lambda p: p[0] * p[1], zip(gamma, net.parameters())))
                
        feat,_ = net(x_spt, vars=backbone, fc_weight=None, fc_bias=None, bn_training=True)
        proto = torch.mean(feat[0:(self.k_spt+5), :], dim=0).unsqueeze(0)
        for c_K in range (self.n_way-1):
            proto = torch.cat((proto, torch.mean(feat[(c_K+1)*(self.k_spt+5):(c_K+2)*(self.k_spt+5), :], dim=0).unsqueeze(0)), dim=0)
        slf_attn = self.SA(proto.unsqueeze(0), proto.unsqueeze(0), proto.unsqueeze(0)).squeeze(0)
        W = 2 * slf_attn / self.temperature
        b = - torch.norm(slf_attn, dim=1) ** 2 / self.temperature
        _,logits = net(x_spt, vars=backbone, fc_weight=W, fc_bias=b, bn_training=True)
        loss = F.cross_entropy(logits, y_spt)
        zero_grad(backbone)
        zero_grad(W)
        zero_grad(b)
        grad = torch.autograd.grad(loss, backbone, retain_graph=True)
        grad_W_b = torch.autograd.grad(loss, (W,b))
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, backbone)))
        fast_W = W - self.update_lr * grad_W_b[0]
        fast_b = b - self.update_lr * grad_W_b[1]

        with torch.no_grad():
            _,logits_q = net(x_qry, backbone, W, b, bn_training=True)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        with torch.no_grad():
            _,logits_q = net(x_qry, fast_weights, fast_W, fast_b, bn_training=True)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            _,logits = net(x_spt, fast_weights, fast_W, fast_b, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            zero_grad(fast_weights)
            zero_grad(fast_W)
            zero_grad(fast_b)
            grad = torch.autograd.grad(loss, fast_weights, retain_graph=True)
            grad_W_b = torch.autograd.grad(loss, (fast_W, fast_b))
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
            fast_W = fast_W - self.update_lr * grad_W_b[0]
            fast_b = fast_b - self.update_lr * grad_W_b[1]
            
            with torch.no_grad():
                _,logits_q = net(x_qry, fast_weights, fast_W, fast_b, bn_training=True)
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()
                corrects[k + 1] = corrects[k + 1] + correct
                
        del net
        accs = np.array(corrects) / querysz
        return accs
    

