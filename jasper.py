# -*- coding: utf-8 -*-
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

import numpy as np

jasper_activations = {
    "hardtanh": nn.Hardtanh,
    "relu": nn.ReLU,
    "selu": nn.SELU,
}


def init_weights(m, mode='xavier_uniform'):
    if isinstance(m, MaskedConv1d):
        init_weights(m.conv, mode)
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        if mode == 'xavier_uniform':
            nn.init.xavier_uniform_(m.weight, gain=1.0)
        elif mode == 'xavier_normal':
            nn.init.xavier_normal_(m.weight, gain=1.0)
        elif mode == 'kaiming_uniform':
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        elif mode == 'kaiming_normal':
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        else:
            raise ValueError("Unknown Initialization mode: {0}".format(mode))
    elif isinstance(m, nn.BatchNorm1d):
        if m.track_running_stats:
            m.running_mean.zero_()
            m.running_var.fill_(1)
            m.num_batches_tracked.zero_()
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def compute_new_kernel_size(kernel_size, kernel_width):
    new_kernel_size = max(int(kernel_size * kernel_width), 1)
    # If kernel is even shape, round up to make it odd
    if new_kernel_size % 2 == 0:
        new_kernel_size += 1
    return new_kernel_size


def get_same_padding(kernel_size, stride, dilation):
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")
    if dilation > 1:
        return (dilation * kernel_size) // 2 - 1
    return kernel_size // 2


class MaskedConv1d(nn.Module):
    __constants__ = ["use_conv_mask", "real_out_channels", "heads"]

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        heads=-1,
        bias=False,
        use_mask=True,
    ):
        super(MaskedConv1d, self).__init__()

        if not (heads == -1 or groups == in_channels):
            raise ValueError("Only use heads for depthwise convolutions")

        self.real_out_channels = out_channels
        if heads != -1:
            in_channels = heads
            out_channels = heads
            groups = heads

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.use_mask = use_mask
        self.heads = heads

    def get_seq_len(self, lens):
        return (
            lens + 2 * self.conv.padding[0] - self.conv.dilation[0] * (self.conv.kernel_size[0] - 1) - 1
        ) / self.conv.stride[0] + 1

    def forward(self, x, lens):
        if self.use_mask:
            lens = lens.to(dtype=torch.long)
            max_len = x.size(2)
            mask = torch.arange(max_len).to(lens.device).expand(len(lens), max_len) >= lens.unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(1).to(device=x.device), 0)
            # del mask
            lens = self.get_seq_len(lens)

        sh = x.shape
        if self.heads != -1:
            x = x.view(-1, self.heads, sh[-1])

        out = self.conv(x)

        if self.heads != -1:
            out = out.view(sh[0], self.real_out_channels, -1)

        return out, lens


class GroupShuffle(nn.Module):
    def __init__(self, groups, channels):
        super(GroupShuffle, self).__init__()

        self.groups = groups
        self.channels_per_group = channels // groups

    def forward(self, x):
        sh = x.shape

        x = x.view(-1, self.groups, self.channels_per_group, sh[-1])

        x = torch.transpose(x, 1, 2).contiguous()

        x = x.view(-1, self.groups * self.channels_per_group, sh[-1])

        return x


class JasperBlock(nn.Module):
    __constants__ = ["conv_mask", "separable", "residual_mode", "res", "mconv"]

    def __init__(
        self,
        inplanes,
        planes,
        repeat=3,
        kernel_size=11,
        kernel_size_factor=1,
        stride=1,
        dilation=1,
        padding='same',
        dropout=0.2,
        activation=None,
        residual=True,
        groups=1,
        separable=False,
        heads=-1,
        normalization="batch",
        norm_groups=1,
        residual_mode='add',
        residual_panes=[],
        conv_mask=False
    ):
        super(JasperBlock, self).__init__()

        if padding != "same":
            raise ValueError("currently only 'same' padding is supported")

        kernel_size_factor = float(kernel_size_factor)
        if type(kernel_size) in (list, tuple):
            kernel_size = [compute_new_kernel_size(k, kernel_size_factor) for k in kernel_size]
        else:
            kernel_size = compute_new_kernel_size(kernel_size, kernel_size_factor)

        padding_val = get_same_padding(kernel_size, stride, dilation)
        self.conv_mask = conv_mask
        self.separable = separable
        self.residual_mode = residual_mode

        inplanes_loop = inplanes
        conv = nn.ModuleList()

        for _ in range(repeat - 1):
            conv.extend(
                self._get_conv_bn_layer(
                    inplanes_loop,
                    planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding_val,
                    groups=groups,
                    heads=heads,
                    separable=separable,
                    normalization=normalization,
                    norm_groups=norm_groups,
                )
            )

            conv.extend(self._get_act_dropout_layer(drop_prob=dropout, activation=activation))


            inplanes_loop = planes

        conv.extend(
            self._get_conv_bn_layer(
                inplanes_loop,
                planes,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding_val,
                groups=groups,
                heads=heads,
                separable=separable,
                normalization=normalization,
                norm_groups=norm_groups,
            )
        )

        self.mconv = conv

        res_panes = residual_panes.copy()
        self.dense_residual = residual

        if residual:
            res_list = nn.ModuleList()
            if len(residual_panes) == 0:
                res_panes = [inplanes]
                self.dense_residual = False
            for ip in res_panes:
                res = nn.ModuleList(
                    self._get_conv_bn_layer(
                        ip, planes, kernel_size=1, normalization=normalization, norm_groups=norm_groups,
                    )
                )

                res_list.append(res)

            self.res = res_list
        else:
            self.res = None

        self.mout = nn.Sequential(*self._get_act_dropout_layer(drop_prob=dropout, activation=activation))

    def _get_conv(
        self,
        in_channels,
        out_channels,
        kernel_size=11,
        stride=1,
        dilation=1,
        padding=0,
        bias=False,
        groups=1,
        heads=-1,
        separable=False,
    ):
        use_mask = self.conv_mask
        if use_mask:
            return MaskedConv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                bias=bias,
                groups=groups,
                heads=heads,
                use_mask=use_mask,
            )
        else:
            return nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                bias=bias,
                groups=groups,
            )

    def _get_conv_bn_layer(
        self,
        in_channels,
        out_channels,
        kernel_size=11,
        stride=1,
        dilation=1,
        padding=0,
        bias=False,
        groups=1,
        heads=-1,
        separable=False,
        normalization="batch",
        norm_groups=1,
    ):
        if norm_groups == -1:
            norm_groups = out_channels

        if separable:
            layers = [
                self._get_conv(
                    in_channels,
                    in_channels,
                    kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding,
                    bias=bias,
                    groups=in_channels,
                    heads=heads,
                ),
                self._get_conv(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    dilation=1,
                    padding=0,
                    bias=bias,
                    groups=groups,
                ),
            ]
        else:
            layers = [
                self._get_conv(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding,
                    bias=bias,
                    groups=groups,
                )
            ]

        if normalization == "group":
            layers.append(nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels))
        elif normalization == "instance":
            layers.append(nn.GroupNorm(num_groups=out_channels, num_channels=out_channels))
        elif normalization == "layer":
            layers.append(nn.GroupNorm(num_groups=1, num_channels=out_channels))
        elif normalization == "batch":
            layers.append(nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1))
        else:
            raise ValueError(
                f"Normalization method ({normalization}) does not match" f" one of [batch, layer, group, instance]."
            )

        if groups > 1:
            layers.append(GroupShuffle(groups, out_channels))
        return layers

    def _get_act_dropout_layer(self, drop_prob=0.2, activation=None):
        if activation is None:
            activation = nn.Hardtanh(min_val=0.0, max_val=20.0)
        layers = [activation, nn.Dropout(p=drop_prob)]
        return layers

    def forward(self, input_: Tuple[List[Tensor], Optional[Tensor]]):
        # type: (Tuple[List[Tensor], Optional[Tensor]]) -> Tuple[List[Tensor], Optional[Tensor]] # nopep8
        lens_orig = None
        xs = input_[0]
        if len(input_) == 2:
            xs, lens_orig = input_
        # compute forward convolutions
        out = xs#[-1]

        lens = lens_orig
        for i, l in enumerate(self.mconv):
            # if we're doing masked convolutions, we need to pass in and
            # possibly update the sequence lengths
            # if (i % 4) == 0 and self.conv_mask:
            if isinstance(l, MaskedConv1d):
                out, lens = l(out, lens)
            else:
                out = l(out)

        # compute the residuals
        if self.res is not None:
            for i, layer in enumerate(self.res):
                res_out = xs#[i]
                for j, res_layer in enumerate(layer):
                    if isinstance(res_layer, MaskedConv1d):
                        res_out, _ = res_layer(res_out, lens_orig)
                    else:
                        res_out = res_layer(res_out)

                if self.residual_mode == 'add':
                    out = out + res_out
                else:
                    out = torch.max(out, res_out)

        # compute the output
        out = self.mout(out)
        if self.res is not None and self.dense_residual:
            return xs + [out], lens

        return out, lens
    
    
class Jasper(nn.Module):
    def __init__(self,labels='abc',audio_conf=None,mid_layers=1,input_size=None):
        super(Jasper,self).__init__()
        self.labels=labels
        self.audio_conf = audio_conf # For consistency with other models
        self.mid_layers = mid_layers
        if not input_size:
            nfft = (self.audio_conf['sample_rate'] * self.audio_conf['window_size'])
            input_size = int(1+(nfft/2))
        self.input_size = input_size
        self.ad_hoc_batch_norm = nn.BatchNorm1d(input_size,track_running_stats=False,affine=False)
        #Jasper blocks created by "JasperEncoder"
        #Bad code, but replicates QuartzNet layout. Need to refactor, a lot.
        blocks = [
                JasperBlock(input_size,256,kernel_size=32,stride=2,dilation=1,residual=False,repeat=1,conv_mask=True,separable=True,activation=torch.nn.ReLU(),dropout=0.2),
                JasperBlock(256,256,kernel_size=32,stride=1,dilation=1,residual=True,repeat=1,conv_mask=True,separable=True,activation=torch.nn.ReLU(),dropout=0.2),
                JasperBlock(256,256,kernel_size=32,stride=1,dilation=1,residual=True,repeat=1,conv_mask=True,separable=True,activation=torch.nn.ReLU(),dropout=0.2),
                JasperBlock(256,256,kernel_size=32,stride=1,dilation=1,residual=True,repeat=1,conv_mask=True,separable=True,activation=torch.nn.ReLU(),dropout=0.2),
                JasperBlock(256,256,kernel_size=38,stride=1,dilation=1,residual=True,repeat=1,conv_mask=True,separable=True,activation=torch.nn.ReLU(),dropout=0.2),
                JasperBlock(256,256,kernel_size=38,stride=1,dilation=1,residual=True,repeat=1,conv_mask=True,separable=True,activation=torch.nn.ReLU(),dropout=0.2),
                JasperBlock(256,256,kernel_size=38,stride=1,dilation=1,residual=True,repeat=1,conv_mask=True,separable=True,activation=torch.nn.ReLU(),dropout=0.2),
                JasperBlock(256,512,kernel_size=50,stride=1,dilation=1,residual=True,repeat=1,conv_mask=True,separable=True,activation=torch.nn.ReLU(),dropout=0.3),
                JasperBlock(512,512,kernel_size=50,stride=1,dilation=1,residual=True,repeat=1,conv_mask=True,separable=True,activation=torch.nn.ReLU(),dropout=0.3),
                JasperBlock(512,512,kernel_size=50,stride=1,dilation=1,residual=True,repeat=1,conv_mask=True,separable=True,activation=torch.nn.ReLU(),dropout=0.3),
                JasperBlock(512,512,kernel_size=62,stride=1,dilation=1,residual=True,repeat=1,conv_mask=True,separable=True,activation=torch.nn.ReLU(),dropout=0.3),
                JasperBlock(512,512,kernel_size=62,stride=1,dilation=1,residual=True,repeat=1,conv_mask=True,separable=True,activation=torch.nn.ReLU(),dropout=0.3),
                JasperBlock(512,512,kernel_size=62,stride=1,dilation=1,residual=True,repeat=1,conv_mask=True,separable=True,activation=torch.nn.ReLU(),dropout=0.4),
                JasperBlock(512,512,kernel_size=74,stride=1,dilation=1,residual=True,repeat=1,conv_mask=True,separable=True,activation=torch.nn.ReLU(),dropout=0.4),
                JasperBlock(512,1024,kernel_size=1,stride=1,dilation=1,residual=False,repeat=1,conv_mask=True,activation=torch.nn.ReLU(),dropout=0.4)
                
                ]
        #self.almost_all_jasper = nn.Sequential(*[zero_block,blocks_123,blocks_456,blocks_7,blocks_89,blocks_101112,blocks_13,blocks_14][:mid_layers])
        self.jasper_encoder = nn.Sequential(*blocks[:mid_layers])
        #Last layer, created by JasperDecoder
        last_layer_input_size = self.jasper_encoder[-1].mconv[-1].num_features
        self.final_layer = nn.Sequential(nn.Conv1d(last_layer_input_size,len(labels),kernel_size=1,stride=1)) #Our labels already include blank
        self.jasper_encoder.apply(init_weights)
        self.final_layer.apply(init_weights)
        
    def forward(self,xs,input_lengths):
        '''
        [Batches X channels X length], lengths
        '''
        xs = self.ad_hoc_batch_norm(xs)
        if np.isinf(xs.cpu()).any() or np.isnan(xs.cpu()).any():
            print('inf or nan after batch norm')
        xs = self.jasper_encoder((xs,input_lengths))[0]
        jasper_res = self.final_layer(xs)
        jasper_res = jasper_res.transpose(2,1) # For consistency with other models.
        if self.training:
            jasper_res = F.log_softmax(jasper_res,dim=-1)
        else:
            jasper_res = F.softmax(jasper_res,dim=-1)
        assert not (jasper_res != jasper_res).any()  # is there any NAN in result?
        return jasper_res # [Batches X Labels X Time (padded to max)]
    
    def get_scaling_factor(self):
        return 2 # this is incorrect, but will work... for now.
    
    @classmethod
    def load_model(cls,path):
        package = torch.load(path,map_location = lambda storage, loc: storage)
        return cls.load_model_package(package)
    
    @classmethod
    def load_model_package(cls,package):
        model = cls(labels=package['labels'],audio_conf=package['audio_conf'],mid_layers=package['layers'],input_size=package.get('input_size'))
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model):
        package = {
                'audio_conf':model.audio_conf,
                'labels':model.labels,
                'layers':model.mid_layers,
                'state_dict':model.state_dict(),
                'input_size':model.input_size
                }
        return package

    