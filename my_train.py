# -*- coding: utf-8 -*-
 
import os
import sys
from collections import namedtuple

import tqdm

from data import label_sets
from wav2letter import Wav2Letter
from jasper import Jasper
from data.data_loader import SpectrogramDataset, BatchAudioDataLoader
import decoder
import torch.optim
import torch.optim.lr_scheduler
from decoder import GreedyDecoder

name_to_model = {
    "jasper":Jasper,
    "wav2letter":Wav2Letter
    }


Wav2LetterLayer = namedtuple('LayerConfig','output_size kernel_size stride dilation dropout')
Wav2LetterConfig = namedtuple('ModelConfig','audio_conf labels mid_layers layers input_size decoder name print_decoded_prob')
AudioConfig = namedtuple('AudioConfig','sample_rate window_size window_stride window')

def get_data_loaders(labels, input_size, audio_conf) :
    train_manifest = 'mini.csv'
    val_manifest = 'mini.csv'
    batch_size = 4

    train_dataset = SpectrogramDataset(train_manifest, audio_conf, labels,mel_spec=input_size)
    train_batch_loader = BatchAudioDataLoader(train_dataset, batch_size=batch_size, num_workers=3)
    eval_dataset = SpectrogramDataset(val_manifest, audio_conf, labels,mel_spec=input_size)
    val_batch_loader = BatchAudioDataLoader(eval_dataset,batch_size=batch_size, num_workers=3)
    return train_batch_loader, val_batch_loader


def get_model(labels, audio_conf):

    layers = [Wav2LetterLayer(256,11,2,1,0.2), Wav2LetterLayer(256,11,1,1,0.2)]
    decoder = GreedyDecoder(labels=labels) 
    my_config = Wav2LetterConfig(audio_conf=audio_conf, labels=labels, mid_layers=len(layers), input_size=64, name='wav2letter', layers=layers, decoder=decoder, print_decoded_prob=0.01)
    
    model = Wav2Letter(my_config)
    return model

def train(model, train_loader, val_loader):
    model.train()
    optimizer= torch.optim.SGD(lr=3e-3, momentum=0.9, nesterov=True, weight_decay=1e-5, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(gamma=1,optimizer=optimizer)
    epochs = 150
    epochs_per_validate = 10
    
    #Standard PyTorch training loop.

    for epoch in tqdm.tqdm(range(1,epochs+1)):
        for batch_idx, batch in enumerate(train_loader):
            inputs, input_lengths, targets, target_lengths, file_paths, texts = batch
            optimizer.zero_grad()
            out, output_lengths = model.forward(inputs,input_lengths)
            loss = model.criterion(out.transpose(0,1), targets, output_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            scheduler.step()

        if epoch % epochs_per_validate == 0:
            validate(model, val_loader)
            model.train()

def validate(model, val_loader):
    model.eval()
    cers = []
    for batch_idx, batch in enumerate(val_loader):
        inputs, input_lengths, targets, target_lengths, file_paths, texts = batch
        out, output_lengths = model.forward(inputs,input_lengths)
        loss = model.criterion(out.transpose(0,1), targets, output_lengths, target_lengths)
        string_metrics = model.add_string_metrics(out, output_lengths, texts, 'valid')
        cers.append(string_metrics['valid_cer'])
        #string_metrics also contains Word error rate and string lengths.
    print(f'Valid CER: {sum(cers)/len(cers)}')


def main():
    labels = label_sets.labels_map['english']
    audio_conf = AudioConfig(sample_rate=16000,window_size=0.02,window_stride=0.01, window='hamming')
    train_loader, val_loader = get_data_loaders(labels, 64, audio_conf)
    model=get_model(labels, audio_conf)
    print(model)
    train(model, train_loader, val_loader)

    

if __name__ == '__main__':
    main()
