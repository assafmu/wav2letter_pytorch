# -*- coding: utf-8 -*-

from __future__ import division

import os
import subprocess

import librosa
import numpy as np
import scipy.signal
from scipy.io import wavfile
import torch
from torch.utils.data import Dataset,DataLoader

windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,'bartlett':scipy.signal.bartlett}

def load_audio(path):
    sr, sound = wavfile.read(path)
    sound = sound.astype('float32') / 32767
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)
    return sound

class SpectrogramDataset(Dataset):
    def __init__(self, audio_conf, manifest_filepath, labels):
        '''
        Some documentation
        '''
        super(SpectrogramDataset, self).__init__()
        with open(manifest_filepath) as f:
            ids = f.readlines()
        self.ids = [x.strip().split(',') for x in ids]
        self.size = len(ids)
        self.window_stride = audio_conf['window_stride']
        self.window_size = audio_conf['window_size']
        self.sample_rate = audio_conf['sample_rate']
        self.window = audio_conf['window']
        self.window = windows.get(audio_conf['window'], windows['hamming'])
        self.labels_map = dict([(labels[i],i) for i in range(len(labels))])
        
    def __getitem__(self, index):
        sample = self.ids[index]
        audio_path, transcript = sample[0], sample[1]
        spect = self.parse_audio(audio_path)
        target = list(filter(None,[self.labels_map.get(x) for x in list(transcript)]))
        return spect, target, audio_path, transcript
    
    def parse_audio(self,audio_path):
        y = load_audio(audio_path)
        n_fft = int(self.sample_rate * self.window_size)
        win_length = n_fft
        hop_length = int(self.sample_rate * self.window_stride)
        
        D = librosa.stft(y, n_fft=n_fft, hop_length = hop_length, win_length=win_length,window=self.window)
        
        spect, phase = librosa.magphase(D)
        
        spect = np.log1p(spect)
        mean = spect.mean()
        std = spect.std()
        spect = np.add(spect,-mean)
        spect = spect / std
        return spect
    
    def __len__(self):
        return self.size

def _collator(batch):
    inputs, targets, file_paths, texts = zip(*batch)
    input_lengths = list(map(lambda input: input.shape[1], inputs))
    target_lengths = list(map(len,targets))
    longest_input = max(input_lengths)
    longest_target = max(target_lengths)
    pad_function = lambda x:np.pad(x,((0,0),(0,longest_input-x.shape[1])),mode='wrap')
    inputs = torch.FloatTensor(list(map(pad_function,inputs)))
    targets = torch.IntTensor([np.pad(np.array(t),(0,longest_target-len(t)),mode='constant') for t in targets])
    return inputs, input_lengths, targets, target_lengths, file_paths, texts

class BatchAudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(BatchAudioDataLoader, self).__init__(*args,**kwargs)
        self.collate_fn = _collator