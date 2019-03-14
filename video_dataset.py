from util import map_data
import torchvision.transforms.functional as F
import random
import itertools
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
import torch


class VideoWrap(object):
    def __init__(self, transform, size):
        self.transform = transform
        self.size = size
    
    def __call__(self, video):
        res = []
        for img in video:
            # pad_w = 0
            # pad_h = 0
            # if img.size[0] < self.size:
            #     pad_w = self.size - img.size[0]
            # if img.size[1] < self.size:
            #     pad_h = self.size - img.size[1]
            
            # if pad_w != 0 or pad_h != 0:
            #     img = transforms.Pad((pad_w//2, pad_h//2), padding_mode='reflect')(img)
            if img.size[0] < img.size[1]:
                img = F.pad(img, (int((1 + img.size[1] - img.size[0]) / 2), 0), padding_mode='reflect')
            if img.size[1] < img.size[0]:
                img = F.pad(img, (0, int((1 + img.size[0] - img.size[1]) / 2)), padding_mode='reflect')

            timg = self.transform(img)
            res.append(timg)
        
        res = torch.stack(res, 0)
        return res


class AFEWDataset(Dataset):
    "Abstract `Dataset` containing images."
    def __init__(self, fns, labels, max_len=120, num_frames_per_clip=16, is_training=True,
                    debug=False, transform=None):

        self.trasform = transform
        self.classes = sorted(list(set(labels)))
        self.class2idx = {v:k for k,v in enumerate(self.classes)}
        self.num_frames_per_clip = num_frames_per_clip
        self.is_training = is_training
        self.max_len = max_len
        self.debug = debug

        file_name = list(map(lambda x: str(x.parent.stem), fns))
        _, idx = np.unique(file_name, return_index=True)
        idx.sort()
        self.slist = fns
        self.sindex = np.append(idx, len(fns))
        
        self.y = np.array([self.class2idx[o] for o in np.asarray(labels)[idx]], dtype=np.int64)
        self.c = len(self.classes)

    def read_select_frames(self, v_idx):
        frames = []
        def generate_index(nframe, idx_start, idx_end, is_training, max_len):

            if is_training:
                if idx_end - idx_start < nframe:
                    idx = random.choices(list(range(idx_start, idx_end)), k=nframe)
                else:
                    idx = random.sample(range(idx_start, idx_end), nframe)
                idx.sort()
            else:
                le = idx_end - idx_start
                if le > max_len: nframe = max_len
                else: nframe = le
                idx = np.linspace(idx_start, idx_end - 1, nframe).astype(int)
            return idx
        
        idx = generate_index(self.num_frames_per_clip, self.sindex[v_idx], self.sindex[v_idx + 1],
                                self.is_training, self.max_len)
        for jj in range(len(idx)):
            frames.append(self.slist[idx[jj]])
        
        frames = sorted(frames, key=lambda x: int(x.stem))
        # frames = sorted(frames, key=lambda x: x.stem)
        
        return frames
    
    def __len__(self):
        if self.debug:
            return 30
        else:
            return len(self.y)

    def __getitem__(self, idx):
        imgs = self.read_select_frames(idx)
        
        assert [i.parent.stem for i in imgs] == [str(imgs[0].parent.stem)]*len(imgs)
        assert imgs[0].parent.parent.stem == self.classes[self.y[idx]]

        imgs = [Image.open(i).convert('RGB') for i in imgs]
        if self.trasform:
            imgs = self.trasform(imgs)
    
        return imgs, self.y[idx]

def image_data(path, train='Train', valid='Valid',
            num_workers=8, size=224, num_frames_per_clip=16,
            bs=32, eval_fact=4, debug=False):

    path=Path(path)
    fn, clss = map_data(root=str(path))

    for p in ['Train', 'Valid']:
        if not debug or p == 'Train': 
            combined = list(zip(fn[p], clss[p]))
            random.shuffle(combined)
            fn[p], clss[p] = zip(*combined)

        fn[p] = list(itertools.chain(*fn[p]))
        clss[p] = list(itertools.chain(*clss[p]))


    tfms = [
        VideoWrap(transforms.Compose([
                    transforms.CenterCrop(size),
                    transforms.RandomRotation(10),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()
                           ]), size=256
                ),

        VideoWrap(transforms.Compose([
                    transforms.CenterCrop(size),
                    transforms.ToTensor()
                           ]), size=256
                )
        
    ] 

    # valid = train if debug else valid
    train_ds = AFEWDataset(fns=fn[train], labels=clss[train], transform=tfms[0],
                                 debug=debug, num_frames_per_clip=num_frames_per_clip)
    valid_ds = AFEWDataset(fns=fn[valid], labels=clss[valid], is_training=False, debug=debug, transform=tfms[1],
                             max_len=eval_fact*bs)
    
    datasets = [train_ds, valid_ds]

    dls = [DataLoader(*o, num_workers=num_workers) for o in
            zip(datasets, (bs, 1), (True, False))] 

    return dls[0], dls[1]