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

class ConvCTCASR(ptl.LightningModule):
    def __init__(self,labels,audio_conf):
        super().__init__()
        self.audio_conf = audio_conf
        self.labels = labels
        
        self.criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        print('LABELS:  '+str(self.labels))
        
    @property 
    def scaling_factor(self):
        raise NotImplementedError()
        
    def forward(inputs,input_lengths):
        raise NotImplementedError()
        #Should return output, output_lengths
        
    #PyTorch Lightning methods
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(),lr=1e-5,momentum=0.9,nesterov=True,weight_decay=1e-5)#Fill this out later
    
    def training_step(self, batch, batch_idx):
        inputs, input_lengths, targets, target_lengths, file_paths, texts = batch
        out, output_lengths = self.forward(inputs,input_lengths)
        loss = self.criterion(out.transpose(0,1),targets,output_lengths,target_lengths)
        self.log_dict({'train_loss':loss})
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, input_lengths, targets, target_lengths, file_paths, texts = batch
        #print(len(texts[0]))
        out, output_lengths = self.forward(inputs,input_lengths)
        #print(targets)
        #print(inputs.shape,input_lengths,targets,target_lengths,out.shape,output_lengths)
        loss = self.criterion(out.transpose(0,1),targets,output_lengths,target_lengths)
        greedy_outputs = out.argmax(dim=-1,keepdim=False)
        
        self.log_dict({'val_loss':loss})
        return loss
