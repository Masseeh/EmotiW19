import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg16, resnet50, resnet18
from advanced.darknet import darknet_53
import os
import numpy as np
from util import load_state_dict
from advanced.layers import Flatten, SelfAttention, ShiftingAttention, Lambda, Att

def apply_stats(x, batch, steps):
    x = x.view(batch, steps, -1)
    mean = x.mean(1)
    std = torch.sqrt( x.var(1) + 1e-16)

    return torch.cat([mean, std], dim=1)

    # return torch.cat([
    # mean.unsqueeze(1),
    # std.unsqueeze(1),
    # maxx.unsqueeze(1),
    # ], dim=1).mean(1)

def softmax_pooling(x, batch, steps):
    x = x.view(batch, -1)
    x = F.softmax(x, dim=1)
    x = x.view(batch, steps, -1)
    return torch.log(x.sum(1)), x

def mean_pooling(x, batch, steps):
    x = x.view(batch, steps, -1)
    x = x.mean(1)        
    return F.log_softmax(x, dim=1), x

pooling_layers = {
    'softmax': softmax_pooling,
    'mean': mean_pooling
}

class Attentive_VGG(nn.Module):
    def __init__(self,
            num_classes,
            attention_hop=0,
            net='vgg16',
            shift=False,
            weights=None,
            dropout=0,
            pooling='softmax'):

        super().__init__()

        self.name = net + "_"

        if attention_hop > 0:
            self.name += "att_"
            self.shift = shift
            self.attention_hop = attention_hop
            # self.attention = ShiftingAttention(512, self.attention_hop, dropout=dropout, shift=self.shift)
            self.attention = SelfAttention(512)
            self.name += f"hop_{self.attention_hop}_{self.shift}_"
            self.use_attention = True
        else:
            self.name += f"avg_"
            self.attention_hop = 1
            self.use_attention = False
            self.attention = nn.AdaptiveAvgPool2d(1)
        
        # self.attention.apply(init_weights)

        self.pooling = pooling
        self.name += pooling

        if self.pooling.find('stats') != -1:
            self.pool_num = 2
        else:
            self.pool_num = 1

        # if self.pooling.find('softmax') != -1:
        #     self.tpooling = Att(dim = 512 * self.attention_hop * self.pool_num)

        self.num_classes = num_classes

        if net.find('vgg16') != -1:
            back = vgg16(False, num_classes=self.num_classes)
            if weights:
                back.load_state_dict(torch.load(os.path.join(weights, 'vgg_to_pytorch.pth')))
                # back.load_state_dict(torch.load(os.path.join(weights, 'vgg_vd_face_fer_dag.pth')))
                
               
            self.features = back.features  

            # self.classifier = nn.Sequential(
            #     Flatten(),
            #     nn.Dropout(inplace=True),
            #     nn.Linear(self.attention_hop * 512, hidden),
            #     nn.Dropout(inplace=True),
            #     nn.Linear(hidden, num_classes)
            # )



            self.classifier = nn.Sequential(
                *back.classifier
            )

            self.attention = Flatten()

            self.linear = nn.Sequential(
                *back.classifier[:-1]
            )

            self.classifier = nn.Sequential(
                # nn.Linear(4096 * 3, self.num_classes)
                nn.Linear(4096, self.num_classes)
            )

            # self.linear = nn.Sequential(
            #     Flatten(),
            #     # nn.LayerNorm(512 * self.attention_hop)
            # )
            
            # self.classifier = nn.Sequential(
            #     nn.Linear(512 * self.attention_hop * self.pool_num, self.num_classes)
            #     # nn.Linear(512 * 3, self.num_classes)
            # )


        elif net.find('resnet50') != -1:
            back = resnet50(False, num_classes=8631)
    
            if weights:
                load_state_dict(back, os.path.join(weights, 'resnet50.pkl')) 

            back = list(back.children())
            model = nn.Sequential(*back)

            self.features = model[:-2]
            self.classifier = nn.Linear(2048 * self.attention_hop * self.pool_num, self.num_classes)
            
            self.linear = Flatten()
        
        elif net.find('darknet') != -1:
            model = darknet_53(self.num_classes)

            self.features = model.layers[:-3]
            self.classifier = nn.Linear(1024 * self.attention_hop * self.pool_num, self.num_classes)
            
            self.linear = Flatten()

        self.dropout = nn.Dropout(dropout, inplace=True)
        print(self.name)

        self.layer_groups = [self.features, self.linear, self.attention, self.classifier]

    def forward(self, x):
        batch, steps, C, H, W = x.size()        
        c_in = x.view(batch * steps, C, H, W)
        x = self.features(c_in) 

        if self.use_attention:    
            x, A = self.attention(x)
        else:
            x = self.attention(x)
            # pass

        x = self.linear(x)

        x = self.dropout(x)

        if self.pooling.find('stats') != -1:    
            x = apply_stats(x, batch, steps)

        # if self.pooling.find('softmax') != -1:
        #     x, temporal_weights = self.tpooling(x.view(batch, steps, -1))
        
        x = self.classifier(x)
        # x = apply_stats(x, batch, steps)

        # res = F.log_softmax(x, dim=1)
        if self.pooling.find('stats') != -1:    
            res = F.log_softmax(x, dim=1)
            temporal_weights = None
        
        # if self.pooling.find('softmax') != -1:
        #     temporal_weights, res = self.tpooling(x.view(batch, steps, -1))

        if self.pooling.find('stats') == -1:    
            res, temporal_weights = pooling_layers[self.pooling](x, batch, steps)
        
        return (res, A, temporal_weights) if self.use_attention else (res, None, temporal_weights)
