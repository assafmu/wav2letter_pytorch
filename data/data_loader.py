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
import pandas as pd

windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,'bartlett':scipy.signal.bartlett}

def load_audio(path):
    sr, sound = wavfile.read(path)
    sound = sound.astype('float32') / (2**15 -1)
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)
    return sound

class SpectrogramDataset(Dataset):
    def __init__(self, manifest_filepath, audio_conf, labels, mel_spec=None, use_cuda=False):
        '''
        Create a dataset for ASR. Audio conf and labels can be re-used from the model.
        Arguments:
            manifest_filepath (string): path to the manifest. Can be a csv containing either path-text pairs, or a pandas DataFrame with columns "filepath" and "text"
            audio_conf (dict): dict containing sample rate, and window size stride and type. 
            labels (list): list containing all valid labels in the text.
            mel_spec(int or None): if not None, use mel spectrogram with that many channels.
            use_cuda(bool): Use torch and torchaudio for stft. Can speed up extraction on GPU.
        '''
        super(SpectrogramDataset, self).__init__()
        prefix_df = pd.read_csv(manifest_filepath,index_col=0,nrows=2)
        if not {'filepath','text'}.issubset(prefix_df.columns):
            self.df = pd.read_csv(manifest_filepath,header=None,names=['filepath','text'])
        else:
            self.df = pd.read_csv(manifest_filepath,index_col=0)
        self.size = len(self.df)
        self.window_stride = audio_conf['window_stride']
        self.window_size = audio_conf['window_size']
        self.sample_rate = audio_conf['sample_rate']
        self.window = audio_conf['window']
        self.window = windows.get(audio_conf['window'], windows['hamming'])
        self.use_cuda = use_cuda
        self.mel_spec = mel_spec
        self.labels_map = dict([(labels[i],i) for i in range(len(labels))])
        self.validate_sample_rate()
        self.preprocess_spectrograms()
        
    
    def preprocess_spectrograms(self):
        self.spects = {}
        for i in range(self.size):
            audio_path = self.df.filepath.iloc[i]
            spect = self.parse_audio(audio_path)
            self.spects[i] = spect
        
    def __getitem__(self, index):
        sample = self.df.iloc[index]
        audio_path, transcript = sample.filepath, sample.text
        if 'א' in self.labels_map: #Hebrew!
            import data.language_specific_tools
            transcript = data.language_specific_tools.hebrew_final_to_normal(transcript)
        #spect = self.parse_audio(audio_path)
        spect = self.spects[index]
        target = list(filter(None,[self.labels_map.get(x) for x in list(transcript)]))
        return spect, target, audio_path, transcript
    
    def _get_spect(self,audio,n_fft,hop_length,win_length):
        if self.use_cuda: # Use torch based convolutions to compute the STFT
            if self.mel_spec:
                import torchaudio
                transform = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate,n_fft=512,n_mels=self.mel_spec)
                return transform(torch.Tensor(audio))
            
            e=torch.stft(torch.FloatTensor(audio),n_fft,hop_length,win_length,window=torch.hamming_window(win_length))
            magnitudes = (e ** 2).sum(dim=2) ** 0.5 
            return magnitudes
        else: # Use CPU bound libraries
            if self.mel_spec:
                import python_speech_features
                spect, energy = python_speech_features.fbank(audio,samplerate=self.sample_rate,winlen=self.window_size,winstep=self.window_stride,winfunc=np.hamming,nfilt=self.mel_spec,nfft=512)
                return spect.T
            D = librosa.stft(audio, n_fft=n_fft, hop_length = hop_length, win_length=win_length,window=scipy.signal.hamming)
            spect, phase = librosa.magphase(D)
    
            return spect
        
    
    def parse_audio(self,audio_path):
        y = load_audio(audio_path)
        n_fft = int(self.sample_rate * self.window_size)
        win_length = n_fft
        hop_length = int(self.sample_rate * self.window_stride)
            
        D = librosa.stft(y, n_fft=n_fft, hop_length = hop_length, win_length=win_length,window=self.window)
        
        spect, phase = librosa.magphase(D)
        spect = self._get_spect(y,n_fft=n_fft,hop_length=hop_length,win_length=win_length)        
        spect = np.log1p(spect)
        if np.isinf(spect).any() or np.isnan(spect).any():
            print('%s spect has INF or NAN before any normalization' % audio_path)
        mean = spect.mean()
        std = spect.std()
        spect = np.add(spect,-mean)
        spect = spect / std
        if np.isinf(spect).any() or np.isnan(spect).any():
            print('%s spect has INF or NAN after spect normalization' % audio_path)
        return spect
    
    def validate_sample_rate(self):
        audio_path = self.df.iloc[0].filepath
        sr,sound = wavfile.read(audio_path)
        assert sr == self.sample_rate, 'Expected sample rate %d but found %d in first file' % (self.sample_rate,sr)
    
    def __len__(self):
        return self.size
    
    def data_channels(self):
        '''
        How many channels are returned in each example.
        '''
        return self.mel_spec or int(1+(int(self.sample_rate * self.window_size)/2))

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
