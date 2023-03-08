# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 11:59:45 2020

@author: Assaf Mushkin
"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import decoder

class ConvCTCASR(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self._cfg = cfg
        self.audio_conf = cfg.audio_conf
        self.labels = cfg.labels
        self.ctc_decoder = cfg.decoder
        self.criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        self.print_decoded_prob = cfg.print_decoded_prob
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
        output_lengths = torch.div(input_lengths, self.scaling_factor, rounding_mode='trunc')
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
            print(f'\nreference: {texts[0]}')
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
    
