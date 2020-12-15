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
import tqdm
import pytorch_lightning
import hydra
from omegaconf import DictConfig, OmegaConf

from data import label_sets, augmentations
from wav2letter import Wav2Letter
from jasper import Jasper
from data.data_loader import SpectrogramDataset, BatchAudioDataLoader
from decoder import GreedyDecoder, PrefixBeamSearchLMDecoder
from torch.utils.data import BatchSampler, SequentialSampler
from novograd import Novograd

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
    #labels = label_sets.labels_map[labels]
    train_dataset = SpectrogramDataset(train_manifest, audio_conf, labels,mel_spec=mel_spec)
    train_batch_loader = BatchAudioDataLoader(train_dataset, batch_size=batch_size)
    eval_dataset = SpectrogramDataset(val_manifest, audio_conf, labels,mel_spec=mel_spec)
    val_batch_loader = BatchAudioDataLoader(eval_dataset,batch_size=batch_size)
    return train_batch_loader, val_batch_loader

@hydra.main(config_name='config')
def main(cfg: DictConfig):
    if type(cfg.model.labels) is str:
        cfg.model.labels = label_sets.labels_map[cfg.model.labels]
    train_loader, val_loader = get_data_loaders(cfg.model.labels,cfg.model.audio_conf,cfg.train_manifest,cfg.val_manifest,4,64) #melspec, batch size, should be derived from config, and melspec should be interpolated
    model = Wav2Letter(cfg.model)
    trainer = pytorch_lightning.Trainer(**cfg.trainer)
    trainer.fit(model,train_loader,val_loader)
    

if __name__ == '__main__':
    main()
