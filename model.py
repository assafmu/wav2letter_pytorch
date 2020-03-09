# -*- coding: utf-8 -*-
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

class InferenceBatchSoftmax(nn.Module):
    def forward(self, input_):
        if not self.training:
            return F.softmax(input_,dim=-1)
        else:
            return input_

class Conv1dBlock(nn.Module):
    def __init__(self,input_size,output_size,kernel_size,stride,drop_out_prob=-1.0,dilation=1,padding='same',bn=True,activation_use=True):
        super(Conv1dBlock, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.drop_out_prob = drop_out_prob
        self.dilation = dilation
        self.activation_use = activation_use
        self.padding = kernel_size[0]
        '''Padding Calculation'''
        input_rows = input_size
        filter_rows = kernel_size[0]
        effective_filter_size_rows = (filter_rows - 1) * dilation + 1
        out_rows = (input_rows + stride - 1) // stride
        self.rows_odd = False
        if padding=='same':
            self.padding_needed = max(0,(out_rows -1) * stride + effective_filter_size_rows - input_rows)
            self.padding_rows = max(0, (out_rows -1) * stride + (filter_rows -1) * dilation + 1 - input_rows)
            self.rows_odd = (self.padding_rows % 2 != 0)
            self.addPaddings = self.padding_rows
        elif padding=='half':
            self.addPaddings = kernel_size[0]
        elif padding=='invalid':
            self.addPaddings = 0
        self.paddingAdded = nn.ReflectionPad1d(self.addPaddings //2) if self.addPaddings > 0 else None
        self.conv1 = nn.Sequential(
                nn.Conv1d(in_channels=input_size,out_channels=output_size,
                          kernel_size=kernel_size,stride=stride,padding=0,dilation=dilation)
        )
        self.batch_norm = nn.BatchNorm1d(num_features=output_size,momentum=0.9,eps=0.001) if bn else None
        self.drop_out = nn.Dropout(drop_out_prob) if self.drop_out_prob != -1 else None

    def forward(self,xs,hid=None):
        if self.paddingAdded is not None:
            xs = self.paddingAdded(xs)
        output = self.conv1(xs)
        if self.batch_norm is not None:
            output = self.batch_norm(output)
        if self.activation_use:
            output = torch.clamp(input=output,min=0,max=20)
        if self.drop_out is not None:
            output = self.drop_out(output)

        return output

class Wav2Letter(nn.Module):
    def __init__(self,labels='abc',audio_conf=None,mid_layers=16):
        super(Wav2Letter,self).__init__()
        self.audio_conf = audio_conf
        self.labels = labels
        self.mid_layers = mid_layers

        nfft = (self.audio_conf['sample_rate'] * self.audio_conf['window_size'])
        input_size = int(1+(nfft/2))

        conv1 = Conv1dBlock(input_size=input_size,output_size=256,kernel_size=(11,),stride=2,dilation=1,drop_out_prob=0.2,padding='same')
        conv2s = []
        conv2s.append(('conv1d_0',conv1))
        layer_size = conv1.output_size
        for idx in range(mid_layers):
            layer_group = idx //3
            if layer_group == 0:
                layer = Conv1dBlock(input_size=layer_size,output_size=256,kernel_size=(11,),stride=1,dilation=1,drop_out_prob=0.2,padding='same')
                conv2s.append(('conv1d_{}'.format(idx+1),layer))
                layer_size = layer.output_size
            elif layer_group == 1:
                layer = Conv1dBlock(input_size=layer_size,output_size=384,kernel_size=(13,),stride=1,dilation=1,drop_out_prob=0.2)
                conv2s.append(('conv1d_{}'.format(idx+1),layer))
                layer_size = layer.output_size
            elif layer_group == 2:
                layer = Conv1dBlock(input_size=layer_size,output_size=512,kernel_size=(17,),stride=1,dilation=1,drop_out_prob=0.2)
                conv2s.append(('conv1d_{}'.format(idx+1),layer))
                layer_size = layer.output_size
            elif layer_group == 3:
                layer = Conv1dBlock(input_size=layer_size,output_size=640,kernel_size=(21,),stride=1,dilation=1,drop_out_prob=0.3)
                conv2s.append(('conv1d_{}'.format(idx+1),layer))
                layer_size = layer.output_size
            elif layer_group == 4:
                layer = Conv1dBlock(input_size=layer_size,output_size=768,kernel_size=(25,),stride=1,dilation=1,drop_out_prob=0.3)
                conv2s.append(('conv1d_{}'.format(idx+1),layer))
                layer_size = layer.output_size
            elif layer_group == 5:
                layer = Conv1dBlock(input_size=layer_size,output_size=896,kernel_size=(29,),stride=1,dilation=2,drop_out_prob=0.4)
                conv2s.append(('conv1d_{}'.format(idx+1),layer))
                layer_size = layer.output_size

        layer = Conv1dBlock(input_size=layer_size, output_size=1024, kernel_size=(1,), stride=1,dilation=1,drop_out_prob=0.4)
        conv2s.append(('conv1d_{}'.format(mid_layers+1),layer))
        layer_size = layer.output_size
        layer = Conv1dBlock(input_size=layer_size, output_size=len(self.labels), kernel_size=(1,), stride=1,bn=False,activation_use=False)
        conv2s.append(('conv1d_{}'.format(mid_layers+2),layer))

        self.conv1ds = nn.Sequential(OrderedDict(conv2s))
        self.inference_softmax = InferenceBatchSoftmax()

    def forward(self, x):
        x = self.conv1ds(x)
        x = x.transpose(1,2)
        x = self.inference_softmax(x)
        return x

    @classmethod
    def load_model(cls,path):
        package = torch.load(path,map_location = lambda storage, loc: storage)
        return cls.load_model_package(package)
    
    @classmethod
    def load_model_package(csl,package):
        model = cls(labels=package['labels'],audio_conf=package['audio_conf'],mid_layers=package['layers'])
        model.load_state_dict(pacakge['state_dict'])
        return model

    @staticmethod
    def serialize(model):
        package = {
                'audio_conf':model.audio_conf,
                'labels':model.labels,
                'mid_layers':model.mid_layers,
                'state_dict':model.state_dict()
                }
        return package

    @staticmethod
    def get_param_size(model):
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp*=x
            params +=tmp
        return params
    