# -*- coding: utf-8 -*-
 
import os
import sys

import pytorch_lightning
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from data import label_sets
from wav2letter import Wav2Letter
from jasper import Jasper
from data.data_loader import SpectrogramDataset, BatchAudioDataLoader

name_to_model = {
    "jasper":Jasper,
    "wav2letter":Wav2Letter
    }
    
def get_data_loaders(labels, cfg):
    train_dataset = SpectrogramDataset(cfg.train_manifest, cfg.audio_conf, labels,mel_spec=cfg.mel_spec)
    train_batch_loader = BatchAudioDataLoader(train_dataset, batch_size=cfg.batch_size)
    eval_dataset = SpectrogramDataset(cfg.val_manifest, cfg.audio_conf, labels,mel_spec=cfg.mel_spec)
    val_batch_loader = BatchAudioDataLoader(eval_dataset,batch_size=cfg.batch_size)
    return train_batch_loader, val_batch_loader

@hydra.main(config_path='configuration', config_name='config')
def main(cfg: DictConfig):
    if type(cfg.model.labels) is str:
        cfg.model.labels = label_sets.labels_map[cfg.model.labels]
    train_loader, val_loader = get_data_loaders(cfg.model.labels,cfg.data)
    model = name_to_model[cfg.model.name](cfg.model)
    trainer = pytorch_lightning.Trainer(**cfg.trainer)
    

    trainer.fit(model, train_loader, val_loader)
    

if __name__ == '__main__':
    main()
