# -*- coding: utf-8 -*-
 
import os
import os.path
import time
import math
import datetime
import random

import librosa
import torch
import torch.nn as nn
import numpy as np
import argparse
import tqdm
import glob
import pytorch_lightning

from data import label_sets, augmentations
from wav2letter import Wav2Letter
from jasper import Jasper
from data.data_loader import SpectrogramDataset, BatchAudioDataLoader
from decoder import GreedyDecoder, PrefixBeamSearchLMDecoder
import timing
from torch.utils.data import BatchSampler, SequentialSampler
from novograd import Novograd

parser = argparse.ArgumentParser(description='Wav2Letter training')
parser.add_argument('--train-manifest',help='path to train manifest csv', default='')
parser.add_argument('--val-manifest',help='path to validation manifest csv',default='')
parser.add_argument('--sample-rate',default=8000,type=int,help='Sample rate')
parser.add_argument('--window-size',default=.02,type=float,help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride',default=.01,type=float,help='Window sstride for spectrogram in seconds')
parser.add_argument('--window',default='hamming',help='Window type for spectrogram generation')
parser.add_argument('--epochs',default=10,type=int,help='Number of training epochs')
parser.add_argument('--lr',default=1e-5,type=float,help='Initial learning rate')
parser.add_argument('--warmup',default=0.2,type=int,help='Percent of steps to warmup learning rate, before cosine annealing. Only used with --lr-sched onecycle')
parser.add_argument('--lr-sched',default='const',type=str,help='Which learning rate scheduler to use. Can be either const, cosine, or onecycle')
parser.add_argument('--batch-size',default=8,type=int,help='Batch size to use during training')
parser.add_argument('--momentum',default=0.9,type=float,help='Momentum')
parser.add_argument('--tensorboard',default='', dest='tensorboard', action='store_true',help='Save tensorboard logs to the specified directory. Defaults to none (no tensorboard logging)')
parser.add_argument('--model-dir',default='',help='Directory to save models. Defaults to none (no models saved)')
parser.add_argument('--name',default='',help='Name to use for tensorboard and model dir, if not specified. Values will be visualize/{name} and models/{name} respectively.')
parser.add_argument('--seed',type=int,default=1234)
parser.add_argument('--layers',default=1,type=int,help='Number of Conv1D blocks, between 1 and 16. 2 Additional last layers are always added.')
parser.add_argument('--labels',default='english',type=str,help='Name of label set to use')
parser.add_argument('--print-samples',default=False,action='store_true',help='Print samples from each epoch')
parser.add_argument('--continue-from',default='',type=str,help='Continue training a saved model')
parser.add_argument('--cuda',default=False,action='store_true',help='Enable training and evaluation with GPU')
parser.add_argument('--epochs-per-save',default=5,type=int,help='How many epochs before saving models')
parser.add_argument('--arc',default='quartz',type=str,help='Network architecture to use. Can be either "quartz" (default) or "wav2letter"')
parser.add_argument('--optimizer',default='sgd',type=str,help='Optimizer to use. can be either "sgd" (default) or "novograd". Note that novograd only accepts --lr parameter.')
parser.add_argument('--mel-spec-count',default=0,type=int,help='How many channels to use in Mel Spectrogram')
parser.add_argument('--use-mel-spec',dest='mel_spec_count',action='store_const',const=64,help='Use mel spectrogram with 64 filters')
parser.add_argument('--augment',default='none',help='Choose augmentation to use. Can be either none (default), specaugment (SpecAugment) or speccutout (SpecCutout)')
parser.add_argument('--default-root-dir', default='.',help='directory for tensorboard logs')

def init_datasets(kwargs):
    labels = label_sets.labels_map[kwargs['labels']]
    audio_conf = get_audio_conf(kwargs)
    train_dataset = SpectrogramDataset(kwargs['train_manifest'], audio_conf, labels,mel_spec=kwargs['mel_spec_count'],use_cuda=kwargs['cuda'])
    batch_sampler = BatchSampler(SequentialSampler(train_dataset), batch_size=kwargs['batch_size'], drop_last=False)
    train_batch_loader = BatchAudioDataLoader(train_dataset, batch_sampler=batch_sampler)
    eval_dataset = SpectrogramDataset(kwargs['val_manifest'], audio_conf, labels,mel_spec=kwargs['mel_spec_count'],use_cuda=kwargs['cuda'])
    return train_dataset, train_batch_loader, eval_dataset
    
def get_optimizer(params,kwargs):
    if kwargs['optimizer'] == 'sgd':
        return torch.optim.SGD(params,lr=kwargs['lr'],momentum=kwargs['momentum'],nesterov=True,weight_decay=1e-5)
    elif kwargs['optimizer'] == 'novograd':
        return Novograd(params,lr=kwargs['lr'])
    return None

def get_scheduler(opt,kwargs,batch_count,epochs):
    total_steps = batch_count * epochs
    if kwargs['lr_sched'] == 'cosine':
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total_steps*(1-kwargs['warmup']), eta_min=0)
        return sched
    if kwargs['lr_sched'] == 'onecycle':
        sched = torch.optim.lr_scheduler.OneCycleLR(opt,10*kwargs['lr'],total_steps=total_steps,pct_start=kwargs['warmup']) # 10 should be configurable.
        return sched
    if kwargs['lr_sched'] != 'const':
        print('Learning rate scheduler %s not found, defaulting to const' % kwargs['lr_sched'])
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda l: 1.0)  # constant

def get_augmentor(kwargs):
    if kwargs['augment'] == 'specaugment':
        return augmentations.SpecAugment()
    if kwargs['augment'] == 'speccutout':
        return augmentations.SpecCutout()
    if kwargs['augment'] != 'none':
        print('Data augmentation %s not found, defaulting to none' % kwargs['augment'])
    return augmentations.Identity()

    
def get_data_loaders(labels,audio_conf,train_manifest,val_manifest,batch_size,mel_spec):
    
    labels = label_sets.labels_map[labels]
    train_dataset = SpectrogramDataset(train_manifest, audio_conf, labels,mel_spec=mel_spec)
    train_batch_loader = BatchAudioDataLoader(train_dataset, batch_size=batch_size)
    eval_dataset = SpectrogramDataset(val_manifest, audio_conf, labels,mel_spec=mel_spec)
    val_batch_loader = BatchAudioDataLoader(eval_dataset,batch_size=batch_size)
    return train_batch_loader, val_batch_loader

def main():
    arguments = parser.parse_args()
    audio_conf = {"window":"hamming","window_stride":0.01,"window_size":0.02,"sample_rate":16000}
    train_loader, val_loader = get_data_loaders('english_lowercase',audio_conf,arguments.train_manifest,arguments.val_manifest,4,64)
    model = Wav2Letter(label_sets.labels_map['english_lowercase'],audio_conf,mid_layers=1,input_size=64)
    trainer = pytorch_lightning.Trainer(default_root_dir=arguments.default_root_dir) # override trainer.default_root_dir with "~/wav2letter_workdir" or something.
    trainer.fit(model,train_loader,val_loader)
    

if __name__ == '__main__':
    main()
