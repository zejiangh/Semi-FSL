import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import sys
    
class CosineLearner(nn.Module):
    def __init__(self, n_way):
        super(CosineLearner, self).__init__()
        
        self.vars = nn.ParameterList()
        fc_w = nn.Parameter(torch.ones(n_way, 640))
        torch.nn.init.kaiming_normal_(fc_w)
        self.vars.append(fc_w)
        if n_way <= 200:
            self.scale_factor = 2
        else:
            self.scale_factor = 10
        
    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        L_norm = torch.norm(vars[0], p=2, dim=1).unsqueeze(1).expand_as(vars[0])
        cos_dist = F.linear(x.div(x_norm + 1e-5), vars[0].div(L_norm + 1e-5))
        logits = self.scale_factor * cos_dist
        return logits
    
    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars
        
        

class ClassifierLearner(nn.Module):
    def __init__(self, n_way):
        super(ClassifierLearner, self).__init__()
        
        self.vars = nn.ParameterList()
        fc_w = nn.Parameter(torch.ones(n_way, 640))
        torch.nn.init.kaiming_normal_(fc_w)
        fc_b = nn.Parameter(torch.zeros(n_way))
        self.vars.append(fc_w)
        self.vars.append(fc_b)
        
    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars
        logits = F.linear(x, vars[0], vars[1])
        return logits
    
    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars
    
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)
    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn
    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)
    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv
        output, attn, log_attn = self.attention(q, k, v)
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output
    
    
class Attenuator(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.size = input_size
        self.fc1 = nn.Linear(self.size, self.size)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(self.size, self.size)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, y):
        out1 = self.fc1(x)
        out1 = self.relu(out1)
        out1 = self.fc2(out1)
        out2 = self.fc1(y)
        out2 = self.relu(out2)
        out2 = self.fc2(out2)
        out = self.sigmoid(out1 + out2)
        return out
    
class SetEncoder(nn.Module):
    def __init__(self):
        super(SetEncoder, self).__init__()
        self.layer1 = self._make_conv2d_layer(3, 32)
        self.layer2 = self._make_conv2d_layer(32, 64)
        self.layer3 = self._make_conv2d_layer(64, 128)
        self.layer4 = self._make_conv2d_layer(128, 256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 48)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    def _make_conv2d_layer(self, in_maps, out_maps):
        return nn.Sequential(
            nn.Conv2d(in_maps, out_maps, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_maps),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False))
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = torch.mean(x, dim=0, keepdim=True)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.sigmoid(self.fc2(x))
        return x
    
    
class ResNet12Learner(nn.Module):
    def __init__(self, n_way, model):
        super(ResNet12Learner, self).__init__()
        
        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList()
        
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                w = nn.Parameter(m.weight.data.detach().cpu().clone(), requires_grad=True)
                self.vars.append(w)
            elif isinstance(m, nn.BatchNorm2d):
                w = nn.Parameter(m.weight.data.detach().cpu().clone(), requires_grad=True)
                b = nn.Parameter(m.bias.data.detach().cpu().clone(), requires_grad=True)
                self.vars.append(w)
                self.vars.append(b)
                r_mean = nn.Parameter(m.running_mean.detach().cpu().clone(), requires_grad=False)
                r_var = nn.Parameter(m.running_var.detach().cpu().clone(), requires_grad=False)
                self.vars_bn.extend([r_mean, r_var])
                
        fc_w = nn.Parameter(torch.ones(n_way, 640), requires_grad=False)
        torch.nn.init.kaiming_normal_(fc_w)
        fc_b = nn.Parameter(torch.zeros(n_way), requires_grad=False)
        self.W = fc_w
        self.b = fc_b
                
    def forward(self, x, vars=None, fc_weight=None, fc_bias=None, bn_training=True):
        if vars is None:
            vars = self.vars
            
        idx = 0
        bn_idx = 0
        
        out = F.conv2d(x, vars[idx], stride=1, padding=1) 
        out = F.batch_norm(out, self.vars_bn[bn_idx], self.vars_bn[bn_idx+1], vars[idx+1], vars[idx+2], training=bn_training)
        out = F.leaky_relu(out, negative_slope=0.1, inplace=True)
        out = F.conv2d(out, vars[idx+3], stride=1, padding=1) 
        out = F.batch_norm(out, self.vars_bn[bn_idx+2], self.vars_bn[bn_idx+3], vars[idx+4], vars[idx+5], training=bn_training)
        out = F.leaky_relu(out, negative_slope=0.1, inplace=True)
        out = F.conv2d(out, vars[idx+6], stride=1, padding=1) 
        out = F.batch_norm(out, self.vars_bn[bn_idx+4], self.vars_bn[bn_idx+5], vars[idx+7], vars[idx+8], training=bn_training)
        residual = F.conv2d(x, vars[idx+9], stride=1)
        residual = F.batch_norm(residual, self.vars_bn[bn_idx+6], self.vars_bn[bn_idx+7], vars[idx+10], vars[idx+11], training=bn_training)
        out = F.leaky_relu(out+residual, negative_slope=0.1, inplace=True)
        b1 = F.max_pool2d(out, kernel_size=2, stride=2, padding=0)
                       
        out = F.conv2d(b1, vars[idx+12], stride=1, padding=1) 
        out = F.batch_norm(out, self.vars_bn[bn_idx+8], self.vars_bn[bn_idx+9], vars[idx+13], vars[idx+14], training=bn_training)
        out = F.leaky_relu(out, negative_slope=0.1, inplace=True)
        out = F.conv2d(out, vars[idx+15], stride=1, padding=1) 
        out = F.batch_norm(out, self.vars_bn[bn_idx+10], self.vars_bn[bn_idx+11], vars[idx+16], vars[idx+17], training=bn_training)
        out = F.leaky_relu(out, negative_slope=0.1, inplace=True)
        out = F.conv2d(out, vars[idx+18], stride=1, padding=1) 
        out = F.batch_norm(out, self.vars_bn[bn_idx+12], self.vars_bn[bn_idx+13], vars[idx+19], vars[idx+20], training=bn_training)
        residual = F.conv2d(b1, vars[idx+21], stride=1)
        residual = F.batch_norm(residual, self.vars_bn[bn_idx+14], self.vars_bn[bn_idx+15], vars[idx+22], vars[idx+23], training=bn_training)
        out = F.leaky_relu(out+residual, negative_slope=0.1, inplace=True)
        b2 = F.max_pool2d(out, kernel_size=2, stride=2, padding=0)
                       
        out = F.conv2d(b2, vars[idx+24], stride=1, padding=1) 
        out = F.batch_norm(out, self.vars_bn[bn_idx+16], self.vars_bn[bn_idx+17], vars[idx+25], vars[idx+26], training=bn_training)
        out = F.leaky_relu(out, negative_slope=0.1, inplace=True)
        out = F.conv2d(out, vars[idx+27], stride=1, padding=1) 
        out = F.batch_norm(out, self.vars_bn[bn_idx+18], self.vars_bn[bn_idx+19], vars[idx+28], vars[idx+29], training=bn_training)
        out = F.leaky_relu(out, negative_slope=0.1, inplace=True)
        out = F.conv2d(out, vars[idx+30], stride=1, padding=1) 
        out = F.batch_norm(out, self.vars_bn[bn_idx+20], self.vars_bn[bn_idx+21], vars[idx+31], vars[idx+32], training=bn_training)
        residual = F.conv2d(b2, vars[idx+33], stride=1)
        residual = F.batch_norm(residual, self.vars_bn[bn_idx+22], self.vars_bn[bn_idx+23], vars[idx+34], vars[idx+35], training=bn_training)
        out = F.leaky_relu(out+residual, negative_slope=0.1, inplace=True)
        b3 = F.max_pool2d(out, kernel_size=2, stride=2, padding=0)
                       
        out = F.conv2d(b3, vars[idx+36], stride=1, padding=1) 
        out = F.batch_norm(out, self.vars_bn[bn_idx+24], self.vars_bn[bn_idx+25], vars[idx+37], vars[idx+38], training=bn_training)
        out = F.leaky_relu(out, negative_slope=0.1, inplace=True)
        out = F.conv2d(out, vars[idx+39], stride=1, padding=1) 
        out = F.batch_norm(out, self.vars_bn[bn_idx+26], self.vars_bn[bn_idx+27], vars[idx+40], vars[idx+41], training=bn_training)
        out = F.leaky_relu(out, negative_slope=0.1, inplace=True)
        out = F.conv2d(out, vars[idx+42], stride=1, padding=1) 
        out = F.batch_norm(out, self.vars_bn[bn_idx+28], self.vars_bn[bn_idx+29], vars[idx+43], vars[idx+44], training=bn_training)
        residual = F.conv2d(b3, vars[idx+45], stride=1)
        residual = F.batch_norm(residual, self.vars_bn[bn_idx+30], self.vars_bn[bn_idx+31], vars[idx+46], vars[idx+47], training=bn_training)
        out = F.leaky_relu(out+residual, negative_slope=0.1, inplace=True)
        b4 = F.max_pool2d(out, kernel_size=2, stride=2, padding=0)
                       
        y = F.avg_pool2d(b4, kernel_size=5, stride=1)
        y = y.view(y.size(0), -1)
        
        if fc_weight is None and fc_bias is None:
            logits = F.linear(y, self.W, self.b)
        else:
            logits = F.linear(y, fc_weight, fc_bias)
        
        return y, logits          
        
    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars
    
    def bn_meanvar(self):
        return self.vars_bn
    

class R12plusLearner(nn.Module):
    '''
    R12plusLearner assumes that final FC-layer was removed
    '''
    def __init__(self, n_way, model):
        super(R12plusLearner, self).__init__()
        
        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList()
        
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                w = nn.Parameter(m.weight.data.detach().cpu().clone())
                self.vars.append(w)
            elif isinstance(m, nn.BatchNorm2d):
                w = nn.Parameter(m.weight.data.detach().cpu().clone())
                b = nn.Parameter(m.bias.data.detach().cpu().clone())
                self.vars.append(w)
                self.vars.append(b)
                r_mean = nn.Parameter(m.running_mean.detach().cpu().clone(), requires_grad=False)
                r_var = nn.Parameter(m.running_var.detach().cpu().clone(), requires_grad=False)
                self.vars_bn.extend([r_mean, r_var])
            elif isinstance(m, nn.Linear):
                w = nn.Parameter(m.weight.data.detach().cpu().clone())
                self.vars.append(w)
                
        fc_w = nn.Parameter(torch.ones(n_way, 640))
        torch.nn.init.kaiming_normal_(fc_w)
        fc_b = nn.Parameter(torch.zeros(n_way))
        self.vars.append(fc_w)
        self.vars.append(fc_b)
                
    def forward(self, x, vars=None, bn_training=True):
        if vars is None:
            vars = self.vars
            
        idx = 0
        bn_idx = 0
        
        out = F.conv2d(x, vars[idx], stride=1, padding=1) 
        out = F.batch_norm(out, self.vars_bn[bn_idx], self.vars_bn[bn_idx+1], vars[idx+1], vars[idx+2], training=bn_training)
        out = F.leaky_relu(out, negative_slope=0.1, inplace=True)
        out = F.conv2d(out, vars[idx+3], stride=1, padding=1) 
        out = F.batch_norm(out, self.vars_bn[bn_idx+2], self.vars_bn[bn_idx+3], vars[idx+4], vars[idx+5], training=bn_training)
        out = F.leaky_relu(out, negative_slope=0.1, inplace=True)
        out = F.conv2d(out, vars[idx+6], stride=1, padding=1) 
        out = F.batch_norm(out, self.vars_bn[bn_idx+4], self.vars_bn[bn_idx+5], vars[idx+7], vars[idx+8], training=bn_training)
        residual = F.conv2d(x, vars[idx+9], stride=1)
        residual = F.batch_norm(residual, self.vars_bn[bn_idx+6], self.vars_bn[bn_idx+7], vars[idx+10], vars[idx+11], training=bn_training)
        b,c,_,_ = out.size()
        attn = F.adaptive_avg_pool2d(out, 1).view(b, c)
        attn = F.relu(F.linear(attn, vars[idx+12]), inplace=True)
        attn = F.sigmoid(F.linear(attn, vars[idx+13])).view(b, c, 1, 1)
        out = F.leaky_relu(out * attn.expand_as(out) + residual, negative_slope=0.1, inplace=True)
        b1 = F.max_pool2d(out, kernel_size=2, stride=2, padding=0)
                       
        out = F.conv2d(b1, vars[idx+14], stride=1, padding=1) 
        out = F.batch_norm(out, self.vars_bn[bn_idx+8], self.vars_bn[bn_idx+9], vars[idx+15], vars[idx+16], training=bn_training)
        out = F.leaky_relu(out, negative_slope=0.1, inplace=True)
        out = F.conv2d(out, vars[idx+17], stride=1, padding=1) 
        out = F.batch_norm(out, self.vars_bn[bn_idx+10], self.vars_bn[bn_idx+11], vars[idx+18], vars[idx+19], training=bn_training)
        out = F.leaky_relu(out, negative_slope=0.1, inplace=True)
        out = F.conv2d(out, vars[idx+20], stride=1, padding=1) 
        out = F.batch_norm(out, self.vars_bn[bn_idx+12], self.vars_bn[bn_idx+13], vars[idx+21], vars[idx+22], training=bn_training)
        residual = F.conv2d(b1, vars[idx+23], stride=1)
        residual = F.batch_norm(residual, self.vars_bn[bn_idx+14], self.vars_bn[bn_idx+15], vars[idx+24], vars[idx+25], training=bn_training)
        b,c,_,_ = out.size()
        attn = F.adaptive_avg_pool2d(out, 1).view(b, c)
        attn = F.relu(F.linear(attn, vars[idx+26]), inplace=True)
        attn = F.sigmoid(F.linear(attn, vars[idx+27])).view(b, c, 1, 1)
        out = F.leaky_relu(out * attn.expand_as(out) + residual, negative_slope=0.1, inplace=True)
        b2 = F.max_pool2d(out, kernel_size=2, stride=2, padding=0)
                       
        out = F.conv2d(b2, vars[idx+28], stride=1, padding=1) 
        out = F.batch_norm(out, self.vars_bn[bn_idx+16], self.vars_bn[bn_idx+17], vars[idx+29], vars[idx+30], training=bn_training)
        out = F.leaky_relu(out, negative_slope=0.1, inplace=True)
        out = F.conv2d(out, vars[idx+31], stride=1, padding=1) 
        out = F.batch_norm(out, self.vars_bn[bn_idx+18], self.vars_bn[bn_idx+19], vars[idx+32], vars[idx+33], training=bn_training)
        out = F.leaky_relu(out, negative_slope=0.1, inplace=True)
        out = F.conv2d(out, vars[idx+34], stride=1, padding=1) 
        out = F.batch_norm(out, self.vars_bn[bn_idx+20], self.vars_bn[bn_idx+21], vars[idx+35], vars[idx+36], training=bn_training)
        residual = F.conv2d(b2, vars[idx+37], stride=1)
        residual = F.batch_norm(residual, self.vars_bn[bn_idx+22], self.vars_bn[bn_idx+23], vars[idx+38], vars[idx+39], training=bn_training)
        b,c,_,_ = out.size()
        attn = F.adaptive_avg_pool2d(out, 1).view(b, c)
        attn = F.relu(F.linear(attn, vars[idx+40]), inplace=True)
        attn = F.sigmoid(F.linear(attn, vars[idx+41])).view(b, c, 1, 1)
        out = F.leaky_relu(out * attn.expand_as(out) + residual, negative_slope=0.1, inplace=True)
        b3 = F.max_pool2d(out, kernel_size=2, stride=2, padding=0)
                       
        out = F.conv2d(b3, vars[idx+42], stride=1, padding=1) 
        out = F.batch_norm(out, self.vars_bn[bn_idx+24], self.vars_bn[bn_idx+25], vars[idx+43], vars[idx+44], training=bn_training)
        out = F.leaky_relu(out, negative_slope=0.1, inplace=True)
        out = F.conv2d(out, vars[idx+45], stride=1, padding=1) 
        out = F.batch_norm(out, self.vars_bn[bn_idx+26], self.vars_bn[bn_idx+27], vars[idx+46], vars[idx+47], training=bn_training)
        out = F.leaky_relu(out, negative_slope=0.1, inplace=True)
        out = F.conv2d(out, vars[idx+48], stride=1, padding=1) 
        out = F.batch_norm(out, self.vars_bn[bn_idx+28], self.vars_bn[bn_idx+29], vars[idx+49], vars[idx+50], training=bn_training)
        residual = F.conv2d(b3, vars[idx+51], stride=1)
        residual = F.batch_norm(residual, self.vars_bn[bn_idx+30], self.vars_bn[bn_idx+31], vars[idx+52], vars[idx+53], training=bn_training)
        b,c,_,_ = out.size()
        attn = F.adaptive_avg_pool2d(out, 1).view(b, c)
        attn = F.relu(F.linear(attn, vars[idx+54]), inplace=True)
        attn = F.sigmoid(F.linear(attn, vars[idx+55])).view(b, c, 1, 1)
        out = F.leaky_relu(out * attn.expand_as(out) + residual, negative_slope=0.1, inplace=True)
        b4 = F.max_pool2d(out, kernel_size=2, stride=2, padding=0)
                       
        y = F.avg_pool2d(b4, kernel_size=5, stride=1)
        y = y.view(y.size(0), -1)
        y = F.linear(y, vars[idx+56], vars[idx+57])
        return y    
        
    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars
    
    def bn_meanvar(self):
        return self.vars_bn
    
    