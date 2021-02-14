# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 11:59:45 2020

@author: Assaf Mushkin
"""
import random

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
        self.print_decoded_prob = cfg.get('print_decoded_prob',0)
        self.example_input_array = self.create_example_input_array()
        
    def create_example_input_array(self):
        batch_size = 4
        min_length,max_length = 100,200
        lengths = torch.randint(min_length,max_length,(4,))
        return (torch.rand(batch_size,self._cfg.input_size,max_length),lengths)
        
    def compute_output_lengths(self,input_lengths):
        '''
        Compute the output lengths given the input lengths.
        Override if ratio is not strictly proportional (can happen with unpadded convolutions)
        '''
        output_lengths = input_lengths // self.scaling_factor
        return output_lengths
    
    @property 
    def scaling_factor(self):
        '''
        Returns a ratio between input lengths and output lengths.
        In convolutional models, depends on kernel size, padding, stride, and dilation.
        '''
        raise NotImplementedError()
        
    def forward(inputs,input_lengths):
        raise NotImplementedError()
        # returns output, output_lengths
        
    def add_string_metrics(self, out, output_lengths, texts, prefix):
        decoded_texts = self.ctc_decoder.decode(out, output_lengths)
        if random.random() < self.print_decoded_prob:
            print(f'reference: {texts[0]}')
            print(f'decoded  : {decoded_texts[0]}')
        wer_sum, cer_sum,wer_denom_sum,cer_denom_sum = 0,0,0,0
        for expected, predicted in zip(texts, decoded_texts):
            cer_value, cer_denom = self.ctc_decoder.cer_ratio(expected, predicted)
            wer_value, wer_denom = self.ctc_decoder.wer_ratio(expected, predicted)
            cer_sum+= cer_value
            cer_denom_sum+=cer_denom
            wer_sum+= wer_value
            wer_denom_sum+=wer_denom
        cer = cer_sum / cer_denom_sum
        wer = wer_sum / wer_denom_sum
        lengths_ratio = sum(map(len, decoded_texts)) / sum(map(len, texts))
        return {prefix+'_cer':cer, prefix+'_wer':wer, prefix+'_len_ratio':lengths_ratio}
        
        
    #PyTorch Lightning methods
    def configure_optimizers(self):
        optimizer = instantiate(self._cfg.optimizer, params=self.parameters())
        scheduler = instantiate(self._cfg.scheduler,optimizer=optimizer)
        return [optimizer],[scheduler]
    
    def training_step(self, batch, batch_idx):
        inputs, input_lengths, targets, target_lengths, file_paths, texts = batch
        out, output_lengths = self.forward(inputs,input_lengths)
        loss = self.criterion(out.transpose(0,1), targets, output_lengths, target_lengths)
        logs = {'train_loss':loss,'learning_rate':self.optimizers().param_groups[0]['lr']}
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
