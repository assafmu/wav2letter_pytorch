# -*- coding: utf-8 -*-
 
import os
import sys
from collections import namedtuple

import pytorch_lightning
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from data import label_sets
from wav2letter import Wav2Letter
from jasper import Jasper
from data.data_loader import SpectrogramDataset, BatchAudioDataLoader
import decoder
import torch.optim
import torch.optim.lr_scheduler
from decoder import GreedyDecoder

name_to_model = {
    "jasper":Jasper,
    "wav2letter":Wav2Letter
    }
    
def get_data_loaders(labels, input_size, audio_conf) :
    #needs labels, batch size, model input size, and the audio config
    train_manifest = 'mini.csv'
    val_manifest = 'mini.csv'
    batch_size = 4

    train_dataset = SpectrogramDataset(train_manifest, audio_conf, labels,mel_spec=input_size)
    train_batch_loader = BatchAudioDataLoader(train_dataset, batch_size=batch_size, num_workers=3)
    eval_dataset = SpectrogramDataset(val_manifest, audio_conf, labels,mel_spec=input_size)
    val_batch_loader = BatchAudioDataLoader(eval_dataset,batch_size=batch_size, num_workers=3)
    return train_batch_loader, val_batch_loader

@hydra.main(config_path='configuration', config_name='config')
def main(cfg: DictConfig):
    #print(cfg.model)
    labels = label_sets.labels_map['english']
    audio_config_template = namedtuple('AudioConfig','sample_rate window_size window_stride window')
    ac = audio_config_template(sample_rate=16000,window_size=0.02,window_stride=0.01, window='hamming')
    #ac = {'sample_rate':16000,'window_size':0.02,'window_stride':0.01,'window':'hamming'}
    w2v_layer_template = namedtuple('LayerConfig','output_size kernel_size stride dilation dropout')
    w2v_model_config_template = namedtuple('ModelConfig','audio_conf labels optimizer scheduler mid_layers layers input_size decoder name print_decoded_prob')
    layers = [w2v_layer_template(256,11,2,1,0.2), w2v_layer_template(256,11,1,1,0.2)]
    decoder = GreedyDecoder(labels=labels)  #namedtuple('DecoderConfig', '_target_ labels')(labels=cfg.model.labels, _target_='decoder.GreedyDecoder')
    scheduler = lambda opt : torch.optim.lr_scheduler.ExponentialLR(gamma=1,optimizer=opt)  #namedtuple('SchedulerConfig', '_target_ gamma')(target='torch.optim.lr_scheduler.ExponentialLR', gamma=1)
    print(labels)
    optimizer= lambda params: torch.optim.SGD(lr=1e-5,momentum=0.9,nesterov=True,weight_decay=1e-5, parameters=params)  #namedtuple('OptimizerConfig', '_target_ 
    my_config = w2v_model_config_template(audio_conf=ac, labels=labels, mid_layers=2, input_size=64, name='wav2letter', layers=layers, decoder=decoder, scheduler=scheduler, optimizer=optimizer, print_decoded_prob=0)
    
    train_loader, val_loader = get_data_loaders(labels, 64, ac)
    model = name_to_model[my_config.name](my_config)
    trainer = pytorch_lightning.Trainer(**cfg.trainer)
    print(model)

    #trainer.fit(model, train_loader, val_loader)
    

if __name__ == '__main__':
    main()
