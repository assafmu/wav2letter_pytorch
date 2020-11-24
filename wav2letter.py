# -*- coding: utf-8 -*-
from collections import OrderedDict
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as ptl
import numpy as np


class Conv1dBlock(nn.Module):
    def __init__(self,input_channels,output_channels,kernel_size,stride,drop_out_prob=-1.0,dilation=1,bn=True,activation_use=True):
        super(Conv1dBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.drop_out_prob = drop_out_prob
        self.dilation = dilation
        self.activation_use = activation_use
        self.padding = kernel_size[0]
        '''Padding Calculation'''
        input_rows = input_channels
        filter_rows = kernel_size[0]
        out_rows = (input_rows + stride - 1) // stride
        self.padding_rows = max(0, (out_rows -1) * stride + (filter_rows -1) * dilation + 1 - input_rows)
        if self.padding_rows > 0:
            if self.padding_rows % 2 == 0:
                self.paddingAdded = nn.ReflectionPad1d(self.padding_rows // 2)
            else:
                self.paddingAdded = nn.ReflectionPad1d((self.padding_rows //2,(self.padding_rows +1)//2))
        else:
            self.paddingAdded =  nn.Identity()
        self.conv1 = nn.Conv1d(in_channels=input_channels,out_channels=output_channels,
                          kernel_size=kernel_size,stride=stride,padding=0,dilation=dilation)
        self.batch_norm = nn.BatchNorm1d(num_features=output_channels,momentum=0.9,eps=0.001) if bn else nn.Identity()
        self.drop_out = nn.Dropout(drop_out_prob) if self.drop_out_prob != -1 else nn.Identity()

    def forward(self,xs):
        xs = self.paddingAdded(xs)
        output = self.conv1(xs)
        output = self.batch_norm(output)
        output = self.drop_out(output)
        if self.activation_use:
            output = torch.clamp(input=output,min=0,max=20)
        return output

class Wav2Letter(ptl.LightningModule):
    def __init__(self,labels='abc',audio_conf=None,mid_layers=1,input_size=None):
        super(Wav2Letter,self).__init__()
        self.audio_conf = audio_conf
        self.labels = labels
        self.mid_layers = mid_layers
        if not input_size:
            nfft = (self.audio_conf['sample_rate'] * self.audio_conf['window_size'])
            input_size = int(1+(nfft/2))
        self.input_size = input_size

        conv1 = Conv1dBlock(input_channels=input_size,output_channels=256,kernel_size=(11,),stride=2,dilation=1,drop_out_prob=0.2)
        conv2s = []
        conv2s.append(('conv1d_0',conv1))
        layer_size = conv1.output_channels
        # Output size, kernel size, stride, dilation, drop_out_prob
        layers = [(256,11,2,1,0.2),
                  (256,11,1,1,0.2), (256,11,1,1,0.2), (256,11,1,1,0.2),
                  (384,13,1,1,0.2), (384,13,1,1,0.2), (384,13,1,1,0.2),
                  (512,17,1,1,0.2), (512,17,1,1,0.2), (512,17,1,1,0.2),
                  (640,21,1,1,0.3), (640,21,1,1,0.3), (640,21,1,1,0.3),
                  (768,25,1,1,0.3), (768,25,1,1,0.3), (768,25,1,1,0.3),
                  (896,29,1,2,0.4), (896,29,1,2,0.4), (896,29,1,2,0.4),
                  ]
        layers = layers[: mid_layers+1] # + 1 for backwards compatability
        layers.append((1024,1,1,1,0.4)) # not inside the list for backwards compatability
        conv_blocks = []
        layer_size = input_size
        for idx in range(len(layers)):
            output_channels, kernel_size, stride, dilation, drop_out_prob = layers[idx]
            layer = Conv1dBlock(input_channels=layer_size,output_channels=output_channels,
                                kernel_size=(kernel_size,),stride=stride,
                                dilation=dilation,drop_out_prob=drop_out_prob)
            layer_size=output_channels
            conv_blocks.append(('conv1d_{}'.format(idx),layer))
        last_layer = Conv1dBlock(input_channels=layer_size, output_channels=len(self.labels), kernel_size=(1,), stride=1,bn=False,activation_use=False)
        conv_blocks.append(('conv1d_{}'.format(len(layers)),last_layer))
        self.conv1ds = nn.Sequential(OrderedDict(conv_blocks))
        strides = []
        for module in self.conv1ds.children():
            strides.append(module.conv1.stride[0])
        self.scaling_factor = int(np.prod(strides))
        self.criterion = nn.CTCLoss(blank=0,reduction='mean')


    def forward(self, x, input_lengths=None):
        x = self.conv1ds(x)
        x = x.transpose(1,2)
        if self.training:
            x = F.log_softmax(x,dim=-1)
        else:
            x = F.softmax(x,dim=-1)
        if input_lengths is not None:
            output_lengths = input_lengths // self.scaling_factor
        else:
            output_lengths = None
        return x, output_lengths
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(),lr=1e-5,momentum=0.9,nesterov=True,weight_decay=1e-5)#Fill this out later
    
    def training_step(self, batch, batch_idx):
        inputs, input_lengths, targets, target_lengths, file_paths, texts = batch
        out, output_lengths = self.forward(inputs,input_lengths)
        return self.criterion(out.transpose(0,1),targets,output_lengths,target_lengths)
    
    def validation_step(self, batch, batch_idx):
        inputs, input_lengths, targets, target_lengths, file_paths, texts = batch
        out, output_lengths = self.forward(inputs,input_lengths)
        return self.criterion(out.transpose(0,1),targets,output_lengths,target_lengths)



