import os
import torch
import torch.nn.functional as F
from video_dataset import image_data
from models import Attentive_VGG
import torchvision
import numpy as np
from fastprogress import master_bar, progress_bar
from copy import deepcopy
import torch.nn.utils as utils
from advanced import one_cycle, cosine_scheduler, adamw
import json
import argparse
import random
from util import accuracy
from advanced.layers import RegularizedLoss

is_cuda = torch.cuda.is_available()
# loss_fn = F.nll_loss
base_path = '/home/masiha/Emotion/Faces'

class WarmStart:
    def __init__(self, optimizer, steps, gamma=0.1, last_epoch=-1):
        self.optimizer = optimizer
        self.base_lrs = np.asarray(list(map(lambda group: group['lr'], self.optimizer.param_groups)))
        self.last_epoch = last_epoch
        self.steps = steps
        self.warm_up = self.steps[0]
        self.gamma = gamma

    def step(self):
        self.last_epoch += 1
        start_lr = self.base_lrs
        if self.last_epoch < self.warm_up:
            start_lr = self.base_lrs*(self.gamma**self.warm_up)
            start_lr *= (1/self.gamma)**self.last_epoch
        
        else:
            for i, step in enumerate(self.steps[1:], 1):
                if self.last_epoch == step:
                    start_lr = self.base_lrs*(self.gamma**i)
        
        print(self.last_epoch + 1 , start_lr)
        for param_group, lr  in zip(self.optimizer.param_groups, start_lr):
            param_group['lr'] = lr

        
def adjust_learning_rate(optimizer, lr, epoch):
    # import ipdb; ipdb.set_trace()
    if epoch < 6:
        start_lr = lr*(0.1**4)
        start_lr *= 10**(epoch-1)
    else:
        start_lr = lr
    
    print(start_lr)

    for param_group, sl  in zip(optimizer.param_groups, start_lr):
        param_group['lr'] = sl

def train(epoch, loss_fn, train_loader, model, optimizer, gclip=5, schd=None):
    model.train()
    loader = progress_bar(train_loader, parent=epoch)

    total_loss = 0

    for data, target in loader:
        if is_cuda:
            data, target = data.cuda(), target.cuda()
            
        optimizer.zero_grad()
        output, att, _ = model(data)
        loss = loss_fn([output, att], target)
        total_loss += loss.item()
        loss.backward()
        if gclip > 0:
            utils.clip_grad_value_(
                model.parameters(),
                gclip
            )
        optimizer.step()

        if schd and hasattr(schd, 'batch_step'):
            schd.batch_step()

        loader.comment = f'Loss: {loss.item():.4f}'
    
    return total_loss / len(train_loader)

def valid(epoch, loss_fn, valid_loader, model):

    model.eval()
    test_loss = 0
    corrects = 0
    sz = 0

    loader = progress_bar(valid_loader, parent=epoch)
    for data, target in loader:       
        if is_cuda:
            data, target = data.cuda(), target.cuda()
        
        output, att, _ = model(data)
        test_loss += loss_fn([output, att], target, reduction='sum').item() # sum up batch loss

        corrects += accuracy(output, target).detach().cpu() * target.size(0)
        sz += target.size(0) 
        
    return test_loss / sz, corrects / sz



def loop(seed, name, ep=30, base_lr=1e-2, lrs=[10, 1], batch_size=3, wd=5e-4, opt='sgd', dropout=0, eval_fact=40,
            net="vgg16", pooling="softmax", debug=True, shift=False,
            attention_hop=0, cuda=0, C=0, **kwargs):

    lr = np.asarray([base_lr/i for i in lrs])

    torch.cuda.set_device(cuda)
    if seed != -1:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    loss_fn = RegularizedLoss(C=C, loss_fn=F.nll_loss)
    
    train_loader, valid_loader = image_data(path=base_path, num_frames_per_clip=16,
                        bs=batch_size, debug=debug, 
                        size=224, num_workers=8, eval_fact=eval_fact)

    model = Attentive_VGG(num_classes=7, shift=shift, weights=os.path.join(base_path, 'models', 'pretrained'), net=net,
                                 pooling=pooling, attention_hop=attention_hop, dropout=dropout)

    if is_cuda:
        model.cuda()
        loss_fn.cuda()

    if opt == 'adam':
        optimizer = adamw.AdamW([{'params': model.layer_groups[0].parameters(), 'lr': lr[0]}],
                                        lr = lr[1], weight_decay=wd, betas=(0.9, 0.99))

        optimizer.add_param_group({'params': model.layer_groups[1].parameters(), 'lr': lr[0]})


    elif opt == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.layer_groups[0].parameters(), 'lr': lr[0]}],
                                        lr = lr[1], weight_decay=wd, momentum=0.9, nesterov=True)

        optimizer.add_param_group({'params': model.layer_groups[1].parameters(), 'lr': lr[0]})

        # optimizer = torch.optim.SGD(model.parameters(),
        #                                 lr = lr[0], weight_decay=wd, momentum=0.9, nesterov=True)


    [optimizer.add_param_group({'params': l.parameters(), 'lr': lr[1]}) for l in model.layer_groups[2:]]

    # schd = None
    # schd = WarmStart(optimizer, [3, 10])
    schd = cosine_scheduler.CosineLRWithRestarts(optimizer, batch_size, len(train_loader.dataset),
                                                        restart_period=10, t_mult=2, verbose=True)

    # schd = one_cycle.OneCycle(optimizer, nb=int(len(train_loader.dataset) * ep /batch_size),
    #                  prcnt=10, div=10)

    best_acc = 0
    best_model = None

    epochs = master_bar(range(1, ep + 1))

    print(f'{"Epoch":^10}|{"TLoss":^10}|{"VLoss":^10}|{"VAccuracy":^10}')

    for epoch in epochs:
        if schd:
            schd.step()
        
        train_loss = train(epochs, loss_fn=loss_fn, train_loader=train_loader,
                                 model=model, optimizer=optimizer, gclip=0, schd=schd)
        if not debug:
            with torch.no_grad():
                valid_loss, acc = valid(epochs, loss_fn=loss_fn, valid_loader=valid_loader, model=model)
        else:
            valid_loss, acc = 0,0
        epochs.write(f'{epoch:^10}|{train_loss:^10.4f}|{valid_loss:^10.4f}|{acc:^10.4f}')

        if acc > best_acc:
            best_acc = acc
            best_model = deepcopy(model.state_dict())
            # torch.save(best_model, os.path.join(base_path, 'models', 'paper', name + '.pth'))
            

    return best_acc

def grid_search():
    from itertools import product

    ats = list(range(2,8))
    Cs = [0, 1e-2, 5e-2, 1e-1, 5e-1, 1]
    best = 0
    best_conf = None

    for at, C in product(ats, Cs):
        print(f'With #attention {at} and C {C}')
        res = []
        for seed in [0, 11, 29]:
            res.append(loop(lr=1e-2, wd=5e-4, seed=seed, opt='adam', cuda=1,
                             C=C, net=torchvision.models.vgg16, pooling='softmax', attention_hidden=[at]))
        
        res = np.asarray(res)
        avg = np.mean(res)
        print(f'attention {at}, C {C}')
        print(f"average accuracy {avg} with std : {np.std(res):.4f}")
        if avg > best:
            best = avg
            best_conf = (at, C)
    
    print(f'best conf {best_conf} with score {best}')
 

def one(exp_name):

    with open('experiments.json') as f:
        exps = json.load(f)

    exp = exps[exp_name]
    print(exp_name, exp)

    res = []
    for seed in exp['seeds']:        

        res.append(loop(**exp, seed=seed, debug=False, name=exp_name))

        print(f"best accuracy {res[-1]}")

    res = np.asarray(res)
    print(f"average accuracy {np.mean(res)} with std : {np.std(res):.4f}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("exp", type=str, help="exp name")
    args = parser.parse_args()

    one(args.exp)
