from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

def frobenius(mat):
    return torch.sum((torch.sum(torch.sum(mat**2, 1), 1) + 1e-10)**0.5)

def conv1d(ni, no, ks=1, stride=1, padding=0, bias=False):
    "Create and iniialize `nn.Conv1d` layer."
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias: conv.bias.data.zero_()
    return nn.utils.spectral_norm(conv)

class SelfAttention(nn.Module):
    "Self attention layer for 2d."
    def __init__(self, dim):
        super().__init__()
        self.query = conv1d(dim, dim//8)
        self.key   = conv1d(dim, dim//8)
        self.value = conv1d(dim, dim)
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def forward(self, x):
        #Notation from https://arxiv.org/pdf/1805.08318.pdf
        size = x.size()
        x = x.view(*size[:2],-1)
        f,g,h = self.query(x),self.key(x),self.value(x)
        beta = F.softmax(torch.bmm(f.permute(0,2,1).contiguous(), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return F.adaptive_avg_pool2d(o.view(*size).contiguous(), 1), beta

def get_fans(shape):
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        fan_in = np.sqrt(np.prod(shape))
        fan_out = np.sqrt(np.prod(shape))
    return fan_in, fan_out

def normal(shape, scale=0.05):
    tensor = torch.FloatTensor(*shape)
    tensor.normal_(mean = 0.0,  std = scale)
    return tensor

def glorot_normal(shape):
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(2. / (fan_in + fan_out))
    return normal(shape, s)

class ShiftingAttention(nn.Module):
    def __init__(self, dim, n, shift=True, dropout=0.5):
        super().__init__()
        self.dim = dim
        self.n_att = n
        self.shift = shift
        
        self.attentions = nn.Sequential( nn.Linear(dim, n, bias=False), nn.Tanh() )
        # self.attentions = nn.Sequential( nn.Linear(dim, 600, bias=True) ,
        #                                  nn.Tanh(),
        #                                  nn.Linear(600, n, bias=False) )

        # self.value = nn.Linear(dim, dim, bias=False)    
        # self.layer_norm = LayerNormalization(dim)


        # self.attentions = nn.Sequential( nn.Conv2d(dim, n, 1, bias=False) , nn.Tanh() )
        # self.attentions = nn.Sequential( nn.Conv2d(dim, 256, 1),
        #                                  nn.Tanh(),
        #                                  nn.Conv2d(256, n, 1, bias=False) )

        # self.attentions = nn.Linear(dim, n)

        self.gnorm = np.sqrt(n)
        
        self.w = nn.Parameter(glorot_normal((n,)))
        self.b = nn.Parameter(glorot_normal((n,)))

    def forward(self, x):
        C, H, W = x.size()[-3:]
        x = x.permute(0, 2, 3, 1)
        x = x.contiguous().view(-1, H*W, C)

        # h = self.value(x)
        '''x = (N, L, F)'''
        scores = self.attentions(x)
        # scores = scores.view(scores.size(0), scores.size(1), -1)
        # x = x.view(x.size(0), x.size(1), -1)
        '''scores = (N, C, L)'''
        weights = F.softmax(scores, dim=1)

        '''weights = (N, C, L), sum(weights, -1) = 1''' 
        if self.shift:       
            outs = []
            for i in range(self.n_att):
                weight = weights[:,:,i]
                ''' weight = (N, L) '''
                weight = weight.unsqueeze(-1).expand_as(x)
                ''' weight = (N, L, F) '''
                
                w = self.w[i].unsqueeze(0).expand(x.size(0), x.size(-1))
                b = self.b[i].unsqueeze(0).expand(x.size(0), x.size(-1))
                ''' center = (N, L, F) '''
                
                o = torch.sum(x * weight, 1).squeeze(1) * w + b
                                
                norm2 = torch.norm(o, 2, -1, keepdim=True).expand_as(o)
                o = o/norm2/self.gnorm
                outs.append(o)
            outputs = torch.cat(outs, -1)        
        else:
            outs = weights.permute(0, 2, 1)@x
            # outs = self.layer_norm(outs)
            weights = weights.permute(0, 2, 1) 
            # outs = x@weights.permute(0, 2, 1)  
            outputs = outs.view(outs.size(0), -1)     
        
        '''outputs = (N, F*C)'''
        return outputs, weights

class Flatten(nn.Module):
    "Flatten `x` to a single dimension, often used at the end of a model. `full` for rank-1 tensor"
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class Lambda(nn.Module):
    "An easy way to create a pytorch layer for a simple `func`."
    def __init__(self, func):
        "create a layer that simply calls `func` with `x`"
        super().__init__()
        self.func=func

    def forward(self, x): return self.func(x)

class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_hid))
        self.beta = nn.Parameter(torch.zeros(d_hid))
        self.eps = eps

    def forward(self, z):
        mean = z.mean(dim=-1, keepdim=True,)
        std = z.std(dim=-1, keepdim=True,)
        ln_out = (z - mean) / (std + self.eps)
        ln_out = self.gamma * ln_out + self.beta

        return ln_out

class Att(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        self.attentions = nn.Sequential( nn.Linear(dim, 1, bias=False))
        
    def forward(self, x):
        '''x = (N, L, F)'''
        scores = self.attentions(x)
        '''scores = (N, C, L)'''
        weights = F.softmax(scores, dim=1)

        outs = weights.permute(0, 2, 1)@x  
        outputs = outs.view(outs.size(0), -1)     
        
        '''outputs = (N, F*C)'''
        return outputs, weights

class RegularizedLoss(nn.Module):  
    def __init__(self, C=0.03, loss_fn=F.nll_loss):
        super().__init__()
        self.C=C
        self.loss_fn=loss_fn

    def forward(self, output, target, reduction='mean'):
        if isinstance(output, (tuple, list)):
            out, att = output[0], output[1]
        else:
            out = output

        if self.C != 0:
            identity = torch.eye(att.size(1), requires_grad=True).cuda()
            identity = identity.unsqueeze(0).expand(att.size(0), att.size(1), att.size(1))
            # penal = frobenius(att@att.transpose(1,2) - identity)/len(att)
            penal = torch.norm(att@att.transpose(1,2) - identity, dim=[-2,-1]).mean() 
        else:
            penal = 0
        
        self.loss2 = penal*self.C
        if reduction == 'sum':
            self.loss1 = self.loss_fn(out, target, reduction='sum')
        else:
            self.loss1 = self.loss_fn(out, target)
        total = self.loss1 + self.loss2 

        return total