# -*- coding: utf-8 -*-
import librosa
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
import glob

from data import label_sets, augmentations
from wav2letter import Wav2Letter
from jasper import Jasper
from data.data_loader import SpectrogramDataset, BatchAudioDataLoader
from decoder import GreedyDecoder, PrefixBeamSearchLMDecoder
import timing
from torch.utils.data import BatchSampler, SequentialSampler
from novograd import Novograd

parser = argparse.ArgumentParser(description='Wav2Letter training')
parser.add_argument('--train-manifest',help='path to train manifest csv', default='')
parser.add_argument('--val-manifest',help='path to validation manifest csv',default='')
parser.add_argument('--sample-rate',default=8000,type=int,help='Sample rate')
parser.add_argument('--window-size',default=.02,type=float,help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride',default=.01,type=float,help='Window sstride for spectrogram in seconds')
parser.add_argument('--window',default='hamming',help='Window type for spectrogram generation')
parser.add_argument('--epochs',default=10,type=int,help='Number of training epochs')
parser.add_argument('--lr',default=1e-5,type=float,help='Initial learning rate')
parser.add_argument('--warmup',default=0.2,type=int,help='Percent of steps to warmup learning rate, before cosine annealing. Only used with --lr-sched onecycle')
parser.add_argument('--lr-sched',default='const',type=str,help='Which learning rate scheduler to use. Can be either const, cosine, or onecycle')
parser.add_argument('--batch-size',default=8,type=int,help='Batch size to use during training')
parser.add_argument('--momentum',default=0.9,type=float,help='Momentum')
parser.add_argument('--tensorboard',default='', dest='tensorboard', action='store_true',help='Save tensorboard logs to the specified directory. Defaults to none (no tensorboard logging)')
parser.add_argument('--model-dir',default='',help='Directory to save models. Defaults to none (no models saved)')
parser.add_argument('--name',default='',help='Name to use for tensorboard and model dir, if not specified. Values will be visualize/{name} and models/{name} respectively.')
parser.add_argument('--seed',type=int,default=1234)
parser.add_argument('--layers',default=1,type=int,help='Number of Conv1D blocks, between 1 and 16. 2 Additional last layers are always added.')
parser.add_argument('--labels',default='english',type=str,help='Name of label set to use')
parser.add_argument('--print-samples',default=False,action='store_true',help='Print samples from each epoch')
parser.add_argument('--continue-from',default='',type=str,help='Continue training a saved model')
parser.add_argument('--cuda',default=False,action='store_true',help='Enable training and evaluation with GPU')
parser.add_argument('--epochs-per-save',default=5,type=int,help='How many epochs before saving models')
parser.add_argument('--arc',default='quartz',type=str,help='Network architecture to use. Can be either "quartz" (default) or "wav2letter"')
parser.add_argument('--optimizer',default='sgd',type=str,help='Optimizer to use. can be either "sgd" (default) or "novograd". Note that novograd only accepts --lr parameter.')
parser.add_argument('--mel-spec-count',default=0,type=int,help='How many channels to use in Mel Spectrogram')
parser.add_argument('--use-mel-spec',dest='mel_spec_count',action='store_const',const=64,help='Use mel spectrogram with 64 filters')
parser.add_argument('--augment',default='none',help='Choose augmentation to use. Can be either none (default), specaugment (SpecAugment) or speccutout (SpecCutout)')


def get_audio_conf(args):
    audio_conf = {k:args[k] for k in ['sample_rate','window_size','window_stride','window']}
    return audio_conf

def init_new_model(arc,channels,kwargs):
    labels = label_sets.labels_map[kwargs['labels']]
    audio_conf = get_audio_conf(kwargs)
    model = arc(labels=labels,audio_conf=audio_conf,mid_layers=kwargs['layers'],input_size=channels)
    return model

def init_model(kwargs,channels):
    arcs_map = {"quartz":Jasper,"wav2letter":Wav2Letter}
    arc = arcs_map[kwargs['arc']]
    if kwargs['continue_from']:
        model = arc.load_model(kwargs['continue_from'])
    else:
        model = init_new_model(arc,channels,kwargs)
    return model

def init_datasets(kwargs):
    labels = label_sets.labels_map[kwargs['labels']]
    audio_conf = get_audio_conf(kwargs)
    train_dataset = SpectrogramDataset(kwargs['train_manifest'], audio_conf, labels,mel_spec=kwargs['mel_spec_count'],use_cuda=kwargs['cuda'])
    batch_sampler = BatchSampler(SequentialSampler(train_dataset), batch_size=kwargs['batch_size'], drop_last=False)
    train_batch_loader = BatchAudioDataLoader(train_dataset, batch_sampler=batch_sampler)
    eval_dataset = SpectrogramDataset(kwargs['val_manifest'], audio_conf, labels,mel_spec=kwargs['mel_spec_count'],use_cuda=kwargs['cuda'])
    return train_dataset, train_batch_loader, eval_dataset
    
def get_optimizer(params,kwargs):
    if kwargs['optimizer'] == 'sgd':
        return torch.optim.SGD(params,lr=kwargs['lr'],momentum=kwargs['momentum'],nesterov=True,weight_decay=1e-5)
    elif kwargs['optimizer'] == 'novograd':
        return Novograd(params,lr=kwargs['lr'])
    return None

def get_scheduler(opt,kwargs,batch_count,epochs):
    total_steps = batch_count * epochs
    if kwargs['lr_sched'] == 'cosine':
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total_steps*(1-kwargs['warmup']), eta_min=0)
        return sched
    if kwargs['lr_sched'] == 'onecycle':
        sched = torch.optim.lr_scheduler.OneCycleLR(opt,10*kwargs['lr'],total_steps=total_steps,pct_start=kwargs['warmup']) # 10 should be configurable.
        return sched
    if kwargs['lr_sched'] != 'const':
        print('Learning rate scheduler %s not found, defaulting to const' % kwargs['lr_sched'])
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda l: 1.0)  # constant

def get_augmentor(kwargs):
    if kwargs['augment'] == 'specaugment':
        return augmentations.SpecAugment()
    if kwargs['augment'] == 'speccutout':
        return augmentations.SpecCutout()
    if kwargs['augment'] != 'none':
        print('Data augmentation %s not found, defaulting to none' % kwargs['augment'])
    return augmentations.Identity()

def train(**kwargs):
    print('starting at %s' % time.asctime())
    train_dataset, train_batch_loader, eval_dataset = init_datasets(kwargs)
    model = init_model(kwargs,train_dataset.data_channels())
    print('Model and datasets initialized')
    if kwargs['name']:
        kwargs['tensorboard'] = kwargs['tensorboard'] or 'visualize/%s' % kwargs['name']
        kwargs['model_dir'] = kwargs['model_dir'] or 'models/%s' % kwargs['name']
    if kwargs['tensorboard']:
        setup_tensorboard(kwargs['tensorboard'])
    training_loop(model,kwargs, train_dataset, train_batch_loader, eval_dataset)   
    if kwargs['tensorboard']:
        _tensorboard_writer.close()

def training_loop(model, kwargs, train_dataset, train_batch_loader, eval_dataset):
    device = 'cuda:0' if torch.cuda.is_available() and kwargs['cuda'] else 'cpu'
    model.to(device)
    greedy_decoder = GreedyDecoder(model.labels)
    criterion = nn.CTCLoss(blank=0,reduction='none')
    parameters = model.parameters()
    optimizer = get_optimizer(parameters,kwargs)
    data_augmentation = get_augmentor(kwargs)
    epochs=kwargs['epochs']
    print('Train dataset size:%d' % len(train_dataset))
    batch_count = math.ceil(len(train_dataset) / kwargs['batch_size'])
    lr_scheduler = get_scheduler(optimizer,kwargs,batch_count,epochs)
    for epoch in range(epochs):
        with timing.EpochTimer(epoch,_log_to_tensorboard) as et:
            model.train()
            total_loss = 0
            for idx, data in et.across_epoch('Data Loading time', tqdm.tqdm(enumerate(train_batch_loader),total=batch_count)):
                inputs, input_lengths, targets, target_lengths, file_paths, texts = data
                with et.timed_action("Data augmetation time"):
                    inputs = data_augmentation(inputs)
                with et.timed_action('Model execution time'):
                    out, output_lengths = model(torch.FloatTensor(inputs).to(device), input_lengths=torch.IntTensor(input_lengths))
                out = out.transpose(1, 0)
                output_lengths = [l // model.scaling_factor for l in input_lengths] # TODO: check types of output_lengths. This computation works, receiving it from model doesn't.
                with et.timed_action('BP time'):
                    loss = criterion(out, targets.to(device), torch.IntTensor(output_lengths), torch.IntTensor(target_lengths))
                    optimizer.zero_grad()
                    loss.mean().backward()
                    optimizer.step()
                    if kwargs['lr_sched'] == 'cosine':
                        lr_scheduler.step((epoch * batch_count + idx) - kwargs['warmup'] * epochs * batch_count)
                    else:
                        lr_scheduler.step(epoch*batch_count + idx)
                total_loss += loss.mean().item()
            log_loss_to_tensorboard(epoch, total_loss / batch_count)
            evaluate(model,eval_dataset,greedy_decoder,epoch,kwargs)
            if epoch != 0 and epoch % kwargs['epochs_per_save'] == 0:
                save_epoch_model(model,epoch, kwargs['model_dir'])
    if kwargs['model_dir']:
        save_model(model, kwargs['model_dir']+'/final.pth')
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
            out, _ = model(torch.FloatTensor(inputs,).unsqueeze(0).to(device), input_lengths=torch.IntTensor([inputs.shape[1]]))
            out_sizes = torch.IntTensor([out.size(1)])
            if idx == index_to_print and kwargs['print_samples']:
                print('Validation case')
                print(text)
                print(''.join(map(lambda i: model.labels[i], torch.argmax(out.transpose(1,0).squeeze(), 1))))
            
            greedy_texts = greedy_decoder.decode(probs=out, sizes=out_sizes)
            greedy_cer[idx] = greedy_decoder.cer_ratio(text, greedy_texts)
            greedy_wer[idx] = greedy_decoder.wer_ratio(text, greedy_texts)
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
    
def save_model(model, path):
    if not path:
        return
    print('saving model to %s' % path)
    os.makedirs(os.path.dirname(path),exist_ok=True)
    package = model.serialize(model)
    torch.save(package, path)
    
def save_epoch_model(model, epoch, path):
    if not path:
        return
    dirname = os.path.splitext(path)[0]
    model_path = os.path.join(dirname,'epoch_%d.pth' % epoch)
    save_model(model, model_path)
    old_files = sorted(glob.glob(dirname+'/epoch_*'),key=os.path.getmtime,reverse=True)[10:]
    for file in old_files:
        os.remove(file)
    
    
if __name__ == '__main__':
    arguments = parser.parse_args()
    train(**vars(arguments))
