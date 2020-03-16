# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import argparse
import tqdm
import os
import os.path
import time
import math
import datetime
import random

from data import label_sets
from model import Wav2Letter
from data.data_loader import SpectrogramDataset, BatchAudioDataLoader
from decoder import GreedyDecoder, PrefixBeamSearchLMDecoder
from torch.utils.data import BatchSampler, SequentialSampler

parser = argparse.ArgumentParser(description='Wav2Letter training')
parser.add_argument('--train-manifest',help='path to train manifest csv', default='data/train.csv')
parser.add_argument('--val-manifest',help='path to validation manifest csv',default='data/validation.csv')
parser.add_argument('--sample-rate',default=8000,type=int,help='Sample rate')
parser.add_argument('--window-size',default=.02,type=float,help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride',default=.01,type=float,help='Window sstride for spectrogram in seconds')
parser.add_argument('--window',default='hamming',help='Window type for spectrogram generation')
parser.add_argument('--epochs',default=10,type=int,help='Number of training epochs')
parser.add_argument('--lr',default=1e-5,type=float,help='Initial learning rate')
parser.add_argument('--batch-size',default=8,type=int,help='Batch size to use during training')
parser.add_argument('--momentum',default=0.9,type=float,help='Momentum')
parser.add_argument('--tensorboard',default=True, dest='tensorboard', action='store_true',help='Turn on tensorboard graphing')
parser.add_argument('--no-tensorboard',dest='tensorboard',action='store_false',help='Turn off tensorboard graphing')
parser.add_argument('--log-dir',default='visualize/wav2letter',type=str,help='Directory for tensorboard logs')
parser.add_argument('--seed',type=int,default=1234)
parser.add_argument('--id',default='Wav2letter training',help='Tensorboard id')
parser.add_argument('--no-model-save',default=False,action='store_true',help='disable model saving entirely')
parser.add_argument('--model-path',default='models/wav2letter.pth.tar')
parser.add_argument('--layers',default=16,type=int,help='Number of Conv1D blocks, between 3 and 16. Last 2 layers are always added.')
parser.add_argument('--labels',default='english',type=str,help='Name of label set to use')
parser.add_argument('--print-samples',default=False,action='store_true',help='Print samples from each epoch')
parser.add_argument('--continue-from',default='',type=str,help='Continue training a saved model')

def get_audio_conf(args):
    audio_conf = {k:args[k] for k in ['sample_rate','window_size','window_stride','window']}
    return audio_conf

def init_new_model(kwargs):
    labels = label_sets.labels_map[kwargs['labels']]
    audio_conf = get_audio_conf(kwargs)
    model = Wav2Letter(labels=labels,audio_conf=audio_conf,mid_layers=kwargs['layers'])
    return model

def init_model(kwargs):
    if kwargs['continue_from']:
        model = Wav2Letter.load_model(kwargs['continue_from'])
    else:
        model = init_new_model(kwargs)
    return model

def init_datasets(audio_conf,labels, kwargs):
    train_dataset = SpectrogramDataset(audio_conf,kwargs['train_manifest'], labels)
    batch_sampler = BatchSampler(SequentialSampler(train_dataset), batch_size=kwargs['batch_size'], drop_last=False)
    train_batch_loader = BatchAudioDataLoader(train_dataset, batch_sampler=batch_sampler)
    eval_dataset = SpectrogramDataset(audio_conf,kwargs['val_manifest'], labels)
    return train_dataset, train_batch_loader, eval_dataset
    

def train(**kwargs):
    print('starting at %s' % time.asctime())
    model = init_model(kwargs)
    train_dataset, train_batch_loader, eval_dataset = init_datasets(model.audio_conf, model.labels, kwargs)
    if kwargs['tensorboard']:
        setup_tensorboard(kwargs['log_dir'])
    training_loop(model,kwargs, train_dataset, train_batch_loader, eval_dataset)    

def training_loop(model, kwargs, train_dataset, train_batch_loader, eval_dataset):
    device = 'cuda:0' if torch.cuda.is_available() and kwargs['cuda'] else 'cpu'
    model.to(device)
    greedy_decoder = GreedyDecoder(model.labels)
    criterion = nn.CTCLoss(blank=0,reduction='none')
    parameters = model.parameters()
    optimizer = torch.optim.SGD(parameters,lr=kwargs['lr'],momentum=kwargs['momentum'],nesterov=True,weight_decay=1e-5)
    scaling_factor = model.get_scaling_factor()
    epochs=kwargs['epochs']
    print('Train dataset size:%d' % len(train_dataset.ids))
    batch_count = math.ceil(len(train_dataset) / kwargs['batch_size'])
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for idx, data in tqdm.tqdm(enumerate(train_batch_loader)):
            inputs, input_lengths, targets, target_lengths, file_paths, texts = data
            out = model(torch.FloatTensor(inputs).to(device))
            out = out.transpose(1,0)
            output_lengths = [l // scaling_factor for l in input_lengths]
            loss = criterion(out, targets.to(device), torch.IntTensor(output_lengths), torch.IntTensor(target_lengths))
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            total_loss += loss.mean().item()
        log_loss_to_tensorboard(epoch, total_loss / batch_count)
        evaluate(model,eval_dataset,greedy_decoder,epoch,kwargs)
    save_model(model, kwargs['model_path'], not kwargs['no_model_save'])
    print('Finished at %s' % time.asctime())
    

def evaluate(model,dataset,greedy_decoder,epoch,kwargs):
    greedy_cer, greedy_wer= compute_error_rates(model, dataset, greedy_decoder, kwargs)
    log_error_rates_to_tensorboard(epoch,greedy_cer.mean(),greedy_wer.mean())
    
def compute_error_rates(model,dataset,greedy_decoder,kwargs):
    device = 'cuda:0' if torch.cuda.is_available() and kwargs['cuda'] else 'cpu'
    model.eval()
    with torch.no_grad():
        num_samples = len(dataset)
        index_to_print = random.randrange(num_samples)
        greedy_cer = np.zeros(num_samples)
        greedy_wer = np.zeros(num_samples)
        for idx, (data) in enumerate(dataset):
            inputs, targets, file_paths, text = data
            out = model(torch.FloatTensor(inputs).unsqueeze(0))
            out = out.transpose(1,0)
            out_sizes = torch.IntTensor([out.size(0)])
            if idx == index_to_print and kwargs['print_samples']:
                print('Validation case')
                print(text)
                print(''.join(map(lambda i: model.labels[i], torch.argmax(out.squeeze(), 1))))
            
            greedy_texts = greedy_decoder.decode(probs=out.transpose(1,0), sizes=out_sizes)
            greedy_cer[idx] = greedy_decoder.cer_ratio(text, greedy_texts[0])
            greedy_wer[idx] = greedy_decoder.wer_ratio(text, greedy_texts[0])
    return greedy_cer, greedy_wer

_tensorboard_writer = None
def setup_tensorboard(log_dir):
    os.makedirs(log_dir,exist_ok=True)
    from tensorboardX import SummaryWriter
    global _tensorboard_writer
    _tensorboard_writer = SummaryWriter(log_dir)
    
def log_loss_to_tensorboard(epoch,avg_loss):
    print('Total loss: %f' % avg_loss)
    _log_to_tensorboard(epoch,{'Avg Train Loss': avg_loss})
    
def log_error_rates_to_tensorboard(epoch,greedy_cer,greedy_wer):
    _log_to_tensorboard(epoch,{'G_CER': greedy_cer, 'G_WER': greedy_wer})
    
def _log_to_tensorboard(epoch,values,tensorboard_id='Wav2Letter training'):
    if _tensorboard_writer:
        _tensorboard_writer.add_scalars(tensorboard_id,values,epoch+1)
    
def save_model(model,path,should_save):
    if not should_save:
        return
    print('saving model to %s' % path)
    os.makedirs(os.path.dirname(path),exist_ok=True)
    package = Wav2Letter.serialize(model)
    torch.save(package, path)
    
if __name__ == '__main__':
    arguments = parser.parse_args()
    train(**vars(arguments))
