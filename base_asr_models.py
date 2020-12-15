# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 11:59:45 2020

@author: Assaf Mushkin
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as ptl
import numpy as np
from hydra.utils import instantiate

class ConvCTCASR(ptl.LightningModule):
    def __init__(self,cfg):
        super().__init__()
        self._cfg = cfg
        self.audio_conf = cfg.audio_conf
        self.labels = cfg.labels
        self.ctc_decoder = instantiate(cfg.decoder)
        self.criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        print('LABELS:  '+str(self.labels))
        
    @property 
    def scaling_factor(self):
        raise NotImplementedError()
        #Should return integer ratio between input length and output length.
        
    def forward(inputs,input_lengths):
        raise NotImplementedError()
        #Should return output, output_lengths
        
    def add_string_metrics(self, out, output_lengths, texts,prefix):
        decoded_texts = self.ctc_decoder.decode(out, output_lengths)
        wer_value, cer_value = 0,0
        for expected, predicted in zip(texts, decoded_texts):
            cer_value += self.ctc_decoder.cer_ratio(expected, predicted)
            wer_value += self.ctc_decoder.wer_ratio(expected, predicted)
        cer_value /= len(texts) #batch size
        wer_value /= len(texts)
        lengths_ratio = sum(map(len, decoded_texts)) / sum(map(len, texts))
        return {prefix+'_cer':cer_value, prefix+'_wer':wer_value, prefix+'_len_ratio':lengths_ratio}
        
        
    #PyTorch Lightning methods
    def configure_optimizers(self):
        return instantiate(self.cfg.optimizer, params=self.parameters()) # add scheduler instantiation. Return tuple of lists
        # return [instantiate(optim...)], [instantiate(scheduler...)]
    
    def training_step(self, batch, batch_idx):
        inputs, input_lengths, targets, target_lengths, file_paths, texts = batch
        out, output_lengths = self.forward(inputs,input_lengths)
        loss = self.criterion(out.transpose(0,1), targets, output_lengths, target_lengths)
        logs = {'train_loss':loss}
        logs.update(self.add_string_metrics(out, output_lengths, texts, 'train'))
        self.log_dict(logs)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, input_lengths, targets, target_lengths, file_paths, texts = batch
        out, output_lengths = self.forward(inputs,input_lengths)
        loss = self.criterion(out.transpose(0,1), targets, output_lengths, target_lengths)
        logs = {'val_loss':loss}
        logs.update(self.add_string_metrics(out, output_lengths, texts, 'val'))
        self.log_dict(logs)
        return loss
