# -*- coding: utf-8 -*-
import torch
import pytest
import numpy as np

from decoder import PrefixBeamSearchLMDecoder, prefix_beam_search, GreedyDecoder
from data.label_sets import english_labels

def greedy_decode(samples, labels, blank_index=0,sizes=None):
    greedy_decoder = GreedyDecoder(labels, blank_index=blank_index)
    res = greedy_decoder.decode(torch.FloatTensor(samples).unsqueeze(0), sizes=sizes)
    return res[0]

def test_sanity():
    sample = np.zeros((10,len(english_labels)))
    sample[0,2] = 0.5
    sample[1,20]=0.5
    sample[2,19]=0.5
    sample[3:,0]=0.5
    res = prefix_beam_search(sample,english_labels)
    assert res == 'ASR'
    
def test_inconsistent_sizes():
    sample = np.zeros((10,len(english_labels) - 1))
    with pytest.raises(AssertionError) as exc_info:
        _ = prefix_beam_search(sample,english_labels)
    assert exc_info is not None
    
    
def test_beam_is_not_greedy():
    '''
    Example from https://towardsdatascience.com/beam-search-decoding-in-ctc-trained-neural-networks-51889a3d85a7
    Shows that beam search can find a path that greedy decoding can not.
    '''
    labels = ['_','A','B',' ']
    samples = np.array([[0.8,0.2,0,0],[0.6,0.4,0,0]])
    res = prefix_beam_search(samples,labels,blank_index=0,return_weights=True)
    assert res == ('A',0.52)
    
    greedy_decoder = GreedyDecoder(labels, blank_index=0)
    greedy_res = greedy_decoder.decode(torch.FloatTensor(samples).unsqueeze(0), sizes=None)
    assert greedy_res == ['']
    
def test_beam_width_changes():
    def the_lm(s):
        if s == 'A':
            return 0.5
        return 1
    
    labels = ['_','A',' ']
    samples = np.array([[0.8,0.2,0],
                        [0.7,0.3,0],
                        [0.6,0.4,0],
                        [0.0,0.0,1]])
    res = prefix_beam_search(samples,labels,lm=the_lm,return_weights=False,k=25,alpha=1,beta=0)
    res2 = prefix_beam_search(samples,labels,lm=the_lm,return_weights=False,k=1,alpha=1,beta=0)
    
    assert res == ' '
    assert res2 == 'A '

def test_class_wrapper():
    
    sample = np.zeros((10,len(english_labels)))
    sample[0,2] = 0.5
    sample[1,20]=0.5
    sample[2,19]=0.5
    sample[3:,0]=0.5
    decoder = PrefixBeamSearchLMDecoder('',english_labels)
    res = decoder.decode(sample)
    assert res == 'ASR'
    
def test_pbs_batch_dimensions():
    sample = torch.zeros((10,len(english_labels)))
    sample[0,2] = 0.5
    sample[1,20]=0.5
    sample[2,19]=0.5
    sample[3:,0]=0.5
    sample = sample.unsqueeze(0)
    decoder = PrefixBeamSearchLMDecoder('',english_labels)
    res = decoder.decode(sample,english_labels)
    assert res == ['ASR']