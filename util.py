import os
from itertools import chain
from pathlib import Path
import numpy as np
# import librosa
import pickle
import torch

local_config = {
            'batch_size': 64, 
            'load_size': 22050*20,
            'phase': 'extract'
            }


def load_state_dict(model, fname):
    """
    Set parameters converted from Caffe models authors of VGGFace2 provide.
    See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.
    Arguments:
        model: model
        fname: file name of parameters converted from a Caffe model, assuming the file format is Pickle.
    """
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')

    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:
            try:
                own_state[name].copy_(torch.from_numpy(param))
            except Exception:
                raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose '\
                                   'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
        else:
            raise KeyError('unexpected key "{}" in state_dict'.format(name))

def map_data(root):

    classes = ('Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise')

    base=Path(root)

    fns = {
        'Train': [],
        'Valid': []
    }

    clss = {
            'Train': [],
            'Valid': []
        }

    for part in ['Train', 'Valid']:
        for c in classes:
            path_to_videos = base/part/c
            for vid in path_to_videos.iterdir():
                frames = sorted(os.listdir(vid), key=lambda x:int(x.split('.')[0]))
                # frames = sorted(os.listdir(vid))
                frames = list(map(lambda x: vid/x, frames))
                fns[part].append(frames)
                clss[part].append([c] * len(frames))

    return fns, clss

def load_from_list(name_list, config=local_config):
    assert len(name_list) == config['batch_size'], \
            "The length of name_list({})[{}] is not the same as batch_size[{}]".format(
                    name_list[0], len(name_list), config['batch_size'])
    audios = np.zeros([config['batch_size'], 1, 1, config['load_size']], dtype=np.float32)
    for idx, audio_path in enumerate(name_list):
        sound_sample, _ = load_audio(audio_path)
        audios[idx] = preprocess(sound_sample, config)
        
    return audios


def load_from_list_extract(name_list, config=local_config):
    audios = []
    audio_paths = []
    for idx, audio_path in enumerate(name_list):
        if idx % 20 is 0:
            print('Processing: {}'.format(idx))
        sound_sample, _ = load_audio(audio_path)
        audios.append(preprocess(sound_sample, config))
        audio_paths.append(audio_path)
        if idx == 2:
            break
    return audios, audio_paths


# NOTE: Load an audio as the same format in soundnet
# 1. Keep original sample rate (which conflicts their own paper)
# 2. Use first channel in multiple channels
# 3. Keep range in [-256, 256]

def load_audio(audio_path, sr=None):
    # By default, librosa will resample the signal to 22050Hz(sr=None). And range in (-1., 1.)
    sound_sample, sr = librosa.load(audio_path, sr=sr, mono=False)

    return sound_sample, sr


def preprocess(raw_audio, config=local_config):
    # Select first channel (mono)
    if len(raw_audio.shape) > 1:
        raw_audio = raw_audio[0]

    # Make range [-256, 256]
    raw_audio *= 256.0

    # Make minimum length available
    length = config['load_size']
    if length > raw_audio.shape[0]:
        raw_audio = np.tile(raw_audio, int(length/raw_audio.shape[0] + 1))

    # Make equal training length
    if config['phase'] != 'extract':
        raw_audio = raw_audio[:length]

    # Check conditions
    assert len(raw_audio.shape) == 1, "It seems this audio contains two channels, we only need the first channel"
    assert np.max(raw_audio) <= 256, "It seems this audio contains signal that exceeds 256"
    assert np.min(raw_audio) >= -256, "It seems this audio contains signal that exceeds -256"

    # Shape to 1 x DIM x 1 x 1
    raw_audio = np.reshape(raw_audio, [1, 1, -1, 1])

    return raw_audio.copy()

def accuracy(input, targs):
    n = targs.shape[0]
    input = input.argmax(dim=-1).view(n,-1)
    targs = targs.view(n,-1)
    return (input==targs).float().mean()
