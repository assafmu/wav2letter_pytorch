# -*- coding: utf-8 -*-

from __future__ import division

import json
import math

import librosa
import numpy as np
import scipy.signal
from scipy.io import wavfile
import soundfile as sf
import torch
import torch.nn
from torch.utils.data import Dataset,DataLoader
import pandas as pd

windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,'bartlett':scipy.signal.bartlett}

def load_audio(path,duration=-1,offset=0):
    with sf.SoundFile(path, 'r') as f:
        dtype = 'float32'
        sample_rate = f.samplerate
        if offset > 0:
            f.seek(int(offset * sample_rate))
        if duration > 0:
            samples = f.read(int(duration * sample_rate), dtype=dtype)
        else:
            samples = f.read(dtype=dtype)
    samples = samples.transpose()
    return samples

class SpectrogramExtractor(torch.nn.Module):
    def __init__(self, audio_conf, mel_spec=64,use_cuda=False):
        super().__init__()
        window_size_samples = int(audio_conf.sample_rate * audio_conf.window_size)
        window_stride_samples = int(audio_conf.sample_rate * audio_conf.window_stride)
        self.n_fft = 2 ** math.ceil(math.log2(window_size_samples))
        filterbanks = torch.tensor(
            librosa.filters.mel(audio_conf.sample_rate,
                                n_fft=self.n_fft,
                                n_mels=mel_spec, fmin=0, fmax=audio_conf.sample_rate / 2),
                                dtype=torch.float
        ).unsqueeze(0)
        self.register_buffer("fb", filterbanks)
        torch_windows = {
                'hann': torch.hann_window,
                'hamming': torch.hamming_window,
                'blackman': torch.blackman_window,
                'bartlett': torch.bartlett_window,
                'none': None,
            }
        window_fn = torch_windows.get(audio_conf.window, None)
        window_tensor = window_fn(window_size_samples, periodic=False) if window_fn else None
        self.register_buffer("window", window_tensor)
        self.stft = lambda x: torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=window_stride_samples,
            win_length=window_size_samples,
            center=True,
            window=self.window.to(dtype=torch.float),
            return_complex=False,
        )
    def _get_spect(self, audio):
        dithering = 1e-5
        preemph = 0.97
        x = torch.Tensor(audio) + torch.randn(audio.shape) * dithering # dithering
        x = torch.cat((x[0].unsqueeze(0), x[1:] - preemph * x[:-1]), dim=0) # preemphasi
        x = self.stft(x.to(device=self.fb.device))
        x = torch.sqrt(x.pow(2).sum(-1)) # get magnitudes
        x = x.pow(2) # power magnitude
        x = torch.matmul(self.fb.to(x.dtype), x) #apply filterbanks
        return x

        
    def extract(self,signal):
        epsilon = 1e-5
        log_zero_guard_value=2 ** -24
        spect = self._get_spect(signal)
        spect = np.log1p(spect + log_zero_guard_value) 
        # normlize across time, per feature
        mean = spect.mean(axis=2)
        std = spect.std(axis=2)
        std += epsilon
        spect = spect - mean.reshape(1, -1, 1)
        spect = spect / std.reshape(1, -1, 1)
        return spect.squeeze()

class SpectrogramDataset(Dataset):
    def __init__(self, manifest_filepath, audio_conf, labels, mel_spec=None, use_cuda=False):
        '''
        Create a dataset for ASR. Audio conf and labels can be re-used from the model.
        Arguments:
            manifest_filepath (string): path to the manifest. Each line must be a json containing fields "audio_filepath" and "text". 
            audio_conf (dict): dict containing sample rate, and window size stride and type. 
            labels (list): list containing all valid labels in the text.
            mel_spec(int or None): if not None, use mel spectrogram with that many channels.
            use_cuda(bool): Use torch and torchaudio for stft. Can speed up extraction on GPU.
        '''
        super(SpectrogramDataset, self).__init__()
        if manifest_filepath.endswith('.csv'):
            self.df = pd.read_csv(manifest_filepath,index_col=0)
        else:
            with open(manifest_filepath) as f:
                lines = f.readlines()
            self.df = pd.DataFrame(map(json.loads,lines))
        if not 'offset' in self.df.columns:
            self.df['offset'] = 0
        if not 'duration' in self.df.columns:
            self.df['duration'] = -1
        self.size = len(self.df)
        self.window_stride = audio_conf['window_stride']
        self.window_size = audio_conf['window_size']
        self.sample_rate = audio_conf['sample_rate']
        self.window = windows.get(audio_conf['window'], windows['hamming'])
        self.use_cuda = use_cuda
        self.mel_spec = mel_spec
        self.labels_map = dict([(labels[i],i) for i in range(len(labels))])
        self.validate_sample_rate()
        self.extractor = SpectrogramExtractor(audio_conf,mel_spec,use_cuda)
        
    def __getitem__(self, index):
        sample = self.df.iloc[index]
        audio_path, transcript = sample.audio_filepath, sample.text
        spect = self.parse_audio(audio_path, sample.duration, sample.offset)
        target = list(filter(None,[self.labels_map.get(x) for x in list(transcript)]))
        return spect, target, audio_path, transcript

    def parse_audio(self,audio_path, duration, offset):
        y = load_audio(audio_path, duration, offset)
        spect = self.extractor.extract(y)
        return spect
    
    def validate_sample_rate(self):
        audio_filepath = self.df.iloc[0].audio_filepath
        sound, sr = sf.read(audio_filepath)
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
    input_lengths = torch.IntTensor(list(map(lambda input: input.shape[1], inputs)))
    target_lengths = torch.IntTensor(list(map(len,targets)))
    longest_input = max(input_lengths).item()
    longest_target = max(target_lengths).item()
    pad_function = lambda x:np.pad(x,((0,0),(0,longest_input-x.shape[1])),mode='constant')
    inputs = torch.FloatTensor(np.array(list(map(pad_function,inputs))))
    targets = torch.IntTensor(np.array([np.pad(np.array(t),(0,longest_target-len(t)),mode='constant') for t in targets]))
    return inputs, input_lengths, targets, target_lengths, file_paths, texts

class BatchAudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(BatchAudioDataLoader, self).__init__(*args,**kwargs)
        self.collate_fn = _collator
