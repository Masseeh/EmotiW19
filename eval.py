import os
import torch
from video_dataset import image_data
from models import Attentive_VGG
import numpy as np
import torch.nn.utils as utils
import json
import argparse
from fastprogress import progress_bar
from util import accuracy

is_cuda = torch.cuda.is_available()
base_path = '/export/livia/Database/AFEW/Faces'


def eval(name, batch_size=3, eval_fact=40, dropout=0,
            net="vgg16", pooling="softmax", shift=False,
            attention_hop=0, **kwargs):

    model = Attentive_VGG(num_classes=7, shift=shift, weights=os.path.join(base_path, 'models', 'pretrained'), net=net,
                                pooling=pooling, attention_hop=attention_hop, dropout=dropout)

    
    model.load_state_dict(torch.load(os.path.join(base_path, 'models', 'paper', name + '.pth'), map_location=lambda storage, loc: storage))

    if is_cuda:
        model.cuda()

    _, valid_loader = image_data(path=base_path, num_frames_per_clip=16,
                        bs=batch_size, debug=False, 
                        size=224, num_workers=8, eval_fact=eval_fact)
    

    model.eval()
    corrects = 0
    sz = 0
    
    with torch.no_grad():

        loader = progress_bar(valid_loader)
        for data, target in loader:       
            if is_cuda:
                data, target = data.cuda(), target.cuda()
            output, spatial, temporal = model(data)
            
            corrects += accuracy(output, target).detach().cpu() * target.size(0)
            sz += target.size(0) 

    print(f'\n{corrects/sz:.4f}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("exp", type=str, help="exp name")
    args = parser.parse_args()

    with open('experiments.json') as f:
        exps = json.load(f)

    exp = exps[args.exp]
    print(args.exp, exp)

    torch.cuda.set_device(0)

    eval(**exp, name=args.exp)
