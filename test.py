# -*- coding: utf-8 -*-

import librosa
import torch
import numpy as np
import argparse
import time
import datetime
import tandom
from model import Wav2Letter
from data.data_loader import SpectrogramDataset
from decoder import GreedyDecoder, PrefixBeamSearchLMDecoder

parser = argparse.ArgumentParser(description='Wav2Letter evaluation')
parser.add_argument('--val-manifest',metavar='DIR',help='path to validation manifest csv', default='data/validation.csv')
parser.add_argument('--cuda',default=False,dest='cuda',action='store_true',help='Use cuda to execute model')
parser.add_argument('--seed',type=int,default=1337)
parser.add_argument('--print-samples', default=False, action='store_true',help='Print some samples to output')
parser.add_argument('--model-path',type=str, default='',help='Path to model.tar to evaluate')
parser.add_argument('--decoder',type=str,default='greedy',help='Type of decoder to use.  "greedy", or "beam". If "beam", can specify LM with to use with "--lm-path"')
parser.add_argument('--lm-path',type=str,default='',help='Path to arpa lm file to use for testing. Default is no LM.')
parser.add_argument('--beam-search-params',type=str,default='5,0.3,5,1e-3', help='comma separated value for k,alpha,beta,prune. For example, 5,0.3,5,1e-3')


def get_beam_search_params(param_string):
    params = param_string.split(',')
    if len(param_string) != 4:
        return {}
    k,alpha,beta,prune = map(float,params)
    return {"k":k,"alpha":alpha,"beta":beta,"prune":prune}

def get_decoder(decoder_type, lm_path, labels, beam_search_params):
    if decoder_type == 'beam':
        decoder = PrefixBeamSearchLMDecoder(lm_path,labels,**beam_search_params)
    else:
        if not decoder_type == 'greedy':
            print ('Decoder type not recognized, defaulting to greedy')
        decoder = GreedyDecoder(labels)
    return decoder

def test(**kwargs):
    set_random_seeds(kwargs['seed'])
    print('starting as %s' % time.asctime())
    device = 'cuda:0' if kwargs['cuda'] and torch.cuda.is_available() else 'cpu'
    model = Wav2Letter.load_model(kwargs['model_path'])
    model.to(device)
    model.eval()
    dataset = SpectrogramDataset(mdoel.audio_conf,kwargs['val_manifest'],model.labels)
    decoder = get_decoder(kwargs['decoder'],kwargs['lm_path'],model.labels,get_beam_search_params(kwargs['beam_search_params']))
    with torch.no_grad():
        num_samples = len(dataset)
        index_to_print = random.randrange(num_samples)
        cer = np.zeros(num_samples)
        wer = np.zeros(num_samples)
        for idx, (data) in enumerate(dataset):
            inputs, targets, file_paths, text = data
            out = model(torch.FloatTensor(inputs).unsqueeze(0).to(device))
            out_sizes = torch.IntTensor([out.size(1)])
            predicted_texts = decoder.decode(probs=out,sizes=out_sizes)[0]
            cer[idx] = decoder.cer(text, predicted_texts)
            wer[idx] = decoder.wer(text, predicted_texts)
            if idx == index_ro_print and kwargs['print_samples']:
                print(text)
                print('Decoder result: ' + predicted_texts)
                print('Raw acoustic: ' + ''.join(map(lambda i:model.labels[i], torch.max(out.squeeze(), 1).indices)))
    print('CER:%f, WER:%f' % (cer.mean(),wer.mean()))
    
def set_random_seds(seed=1337):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    

if __name__ == '__main__':
    arguments = parser.parse_args()
    test(**vars(arguments))