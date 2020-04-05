# -*- coding: utf-8 -*-
import librosa
import torch
import numpy as np
import argparse
import time
import random
from wav2letter import Wav2Letter
from jasper import Jasper
from data.data_loader import SpectrogramDataset
from decoder import GreedyDecoder, PrefixBeamSearchLMDecoder

parser = argparse.ArgumentParser(description='Wav2Letter evaluation')
parser.add_argument('--test-manifest',metavar='DIR',help='path to test manifest csv', default='data/test.csv')
parser.add_argument('--cuda',default=False,dest='cuda',action='store_true',help='Use cuda to execute model')
parser.add_argument('--seed',type=int,default=1337)
parser.add_argument('--print-samples', default=False, action='store_true',help='Print some samples to output')
parser.add_argument('--print-all',default=False,action='store_true',help='Print all samples to output')
parser.add_argument('--model-path',type=str, default='',help='Path to model.tar to evaluate')
parser.add_argument('--decoder',type=str,default='greedy',help='Type of decoder to use.  "greedy", or "beam". If "beam", can specify LM with to use with "--lm-path"')
parser.add_argument('--lm-path',type=str,default='',help='Path to arpa lm file to use for testing. Default is no LM.')
parser.add_argument('--beam-search-params',type=str,default='5,0.3,5,1e-3', help='comma separated value for k,alpha,beta,prune. For example, 5,0.3,5,1e-3')
parser.add_argument('--arc',default='quartz',type=str,help='Network architecture to use. Can be either "quartz" (default) or "wav2letter"')
parser.add_argument('--mel-spec-count',default=0,type=int,help='How many channels to use in Mel Spectrogram')
parser.add_argument('--use-mel-spec',dest='mel_spec_count',action='store_const',const=64,help='Use mel spectrogram with default value (64)')

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

def get_model(kwargs):
    arcs_map = {"quartz":Jasper,"wav2letter":Wav2Letter}
    arc = arcs_map[kwargs['arc']]
    model = arc.load_model(kwargs['model_path'])
    return model
    
def test(**kwargs):
    set_random_seeds(kwargs['seed'])
    print('starting as %s' % time.asctime())
    device = 'cuda:0' if kwargs['cuda'] and torch.cuda.is_available() else 'cpu'
    model = get_model(kwargs)
    model.to(device)
    model.eval()
    dataset = SpectrogramDataset(kwargs['test_manifest'], model.audio_conf, model.labels, mel_spec=kwargs['mel_spec_count'], use_cuda=kwargs['cuda'])
    decoder = get_decoder(kwargs['decoder'],kwargs['lm_path'], model.labels, get_beam_search_params(kwargs['beam_search_params']))
    with torch.no_grad():
        num_samples = len(dataset)
        index_to_print = random.randrange(num_samples)
        cer = np.zeros(num_samples)
        wer = np.zeros(num_samples)
        for idx, (data) in enumerate(dataset):
            inputs, targets, file_paths, text = data
            out = model(torch.FloatTensor(inputs).unsqueeze(0).to(device), input_lengths=torch.IntTensor([inputs.shape[1]]))
            out_sizes = torch.IntTensor([out.size(1)])
            predicted_texts = decoder.decode(probs=out,sizes=out_sizes)[0]
            cer[idx] = decoder.cer_ratio(text, predicted_texts)
            wer[idx] = decoder.wer_ratio(text, predicted_texts)
            if (idx == index_to_print and kwargs['print_samples']) or kwargs['print_all']:
                print(text)
                print('Decoder result: ' + predicted_texts)
                print('Raw acoustic: ' + ''.join(map(lambda i: model.labels[i], torch.argmax(out.squeeze(), 1))))
    print('CER:%f, WER:%f' % (cer.mean(),wer.mean()))
    
def set_random_seeds(seed=1337):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    

if __name__ == '__main__':
    arguments = parser.parse_args()
    test(**vars(arguments))