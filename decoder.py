#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
# Modified to support pytorch Tensors


import torch
from six.moves import xrange
import Levenshtein as Lev
import re
from collections import defaultdict, Counter
import numpy as np

class Decoder(object):
    """
    Basic decoder class from which all other decoders inherit. Implements several
    helper functions. Subclasses should implement the decode() method.

    Arguments:
        labels (string): mapping from integers to characters.
        blank_index (int, optional): index for the blank '_' character. Defaults to 0.
        space_index (int, optional): index for the space ' ' character. Defaults to 28.
    """

    def __init__(self, labels, blank_index=0):
        # e.g. labels = "_'ABCDEFGHIJKLMNOPQRSTUVWXYZ#"
        self.labels = labels
        self.int_to_char = dict([(i, c) for (i, c) in enumerate(labels)])
        self.blank_index = blank_index
        space_index = len(labels)  # To prevent errors in decode, we add an out of bounds index for the space
        if ' ' in labels:
            space_index = labels.index(' ')
        self.space_index = space_index

    def wer(self, s1, s2):
        """
        Computes the Word Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to words.
        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """

        # build mapping of words to integers
        b = set(s1.split() + s2.split())
        word2char = dict(zip(b, range(len(b))))

        # map the words to a char array (Levenshtein packages only accepts
        # strings)
        w1 = [chr(word2char[w]) for w in s1.split()]
        w2 = [chr(word2char[w]) for w in s2.split()]

        return Lev.distance(''.join(w1), ''.join(w2))

    def cer(self, s1, s2):
        """
        Computes the Character Error Rate, defined as the edit distance.

        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """
        s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
        return Lev.distance(s1, s2)
    
    def cer_ratio(self, expected, predicted):
        return 100 * self.cer(expected,predicted) / float(len(expected))
    
    def wer_ratio(self, expected, predicted):
        return 100 * self.wer(expected,predicted) / float(len(expected.split()))
    
    def decode(self, probs, sizes=None):
        """
        Given a matrix of character probabilities, returns the decoder's
        best guess of the transcription

        Arguments:
            probs: Tensor of character probabilities, where probs[c,t]
                            is the probability of character c at time t
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            string: sequence of the model's best guess for the transcription
        """
        raise NotImplementedError


class GreedyDecoder(Decoder):
    def __init__(self, labels, blank_index=0):
        super(GreedyDecoder, self).__init__(labels, blank_index)

    def convert_to_strings(self, sequences, sizes=None, remove_repetitions=False, return_offsets=False):
        """Given a list of numeric sequences, returns the corresponding strings"""
        strings = []
        offsets = [] if return_offsets else None
        for x in xrange(len(sequences)):
            seq_len = sizes[x] if sizes is not None else len(sequences[x])
            string, string_offsets = self.process_string(sequences[x], seq_len, remove_repetitions)
            strings.append([string])  # We only return one path
            if return_offsets:
                offsets.append([string_offsets])
        if return_offsets:
            return strings, offsets
        else:
            return strings

    def process_string(self, sequence, size, remove_repetitions=False):
        string = ''
        offsets = []
        for i in range(size):
            char = self.int_to_char[sequence[i].item()]
            if char != self.int_to_char[self.blank_index]:
                # if this char is a repetition and remove_repetitions=true, then skip
                if remove_repetitions and i != 0 and char == self.int_to_char[sequence[i - 1].item()]:
                    pass
                elif char == self.labels[self.space_index]:
                    string += ' '
                    offsets.append(i)
                else:
                    string = string + char
                    offsets.append(i)
        return string, torch.IntTensor(offsets)

    def decode(self, probs, sizes=None):
        """
        Returns the argmax decoding given the probability matrix. Removes
        repeated elements in the sequence, as well as blanks.

        Arguments:
            probs: Tensor of character probabilities from the network. Expected shape of batch x seq_length x output_dim
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            strings: sequences of the model's best guess for the transcription on inputs
            offsets: time step per character predicted
        """
        _, max_probs = torch.max(probs, 2)
        strings, offsets = self.convert_to_strings(max_probs.view(max_probs.size(0), max_probs.size(1)), sizes,
                                                   remove_repetitions=True, return_offsets=True)
        return strings[0]
    
def prefix_beam_search(ctc, labels, blank_index=0, lm=None,k=5,alpha=0.3,beta=5,prune=0.001,end_char='>',return_weights=False):
    """
    Performs prefix beam search on the output of a CTC network.
    Originally from https://github.com/corticph/prefix-beam-search, with minor edits.
    Args:
        ctc (np.ndarray): The CTC output. Should be a 2D array (timesteps x alphabet_size)
        lm (func): Language model function. Should take as input a string and output a probability.
        k (int): The beam width. Will keep the 'k' most likely candidates at each timestep.
        alpha (float): The language model weight. Should usually be between 0 and 1.
        beta (float): The language model compensation term. The higher the 'alpha', the higher the 'beta'.
        prune (float): Only extend prefixes with chars with an emission probability higher than 'prune'.
        return_weights(bool): return the confidence of the decoded string.
    Returns:
        string: The decoded CTC output.
    """
    assert (ctc.shape[1] == len(labels)), "ctc size:%d, labels: %d" % (ctc.shape[1], len(labels))
    assert ctc.shape[0] > 1, "ctc length: %d was too short" % ctc.shape[0]
    assert (ctc > 0).all(), 'ctc output contains negative numbers'
    lm = (lambda l: 1) if lm is None else lm # if no LM is provided, just set to function returning 1
    word_count_re = re.compile(r'\w+[\s|>]')
    W = lambda l: word_count_re.findall(l)
    F = ctc.shape[1]
    
    ctc = np.vstack((np.zeros(F), ctc)) # just add an imaginative zero'th step (will make indexing more intuitive)
    T = ctc.shape[0]
    blank_char = labels[blank_index]

    # STEP 1: Initiliazation
    O = ''
    Pb, Pnb = defaultdict(Counter), defaultdict(Counter)
    Pb[0][O] = 1
    Pnb[0][O] = 0
    A_prev = [O]
    # END: STEP 1

    # STEP 2: Iterations and pruning
    for t in range(1, T):
        pruned_alphabet = [labels[i] for i in np.where(ctc[t] > prune)[0]]
        for l in A_prev:
            
            if len(l) > 0 and l[-1] == end_char:
                Pb[t][l] = Pb[t - 1][l]
                Pnb[t][l] = Pnb[t - 1][l]
                continue  

            for c in pruned_alphabet:
                c_ix = labels.index(c)
                # END: STEP 2
                
                # STEP 3: “Extending” with a blank
                if c == blank_char:
                    Pb[t][l] += ctc[t][blank_index] * (Pb[t - 1][l] + Pnb[t - 1][l])
                # END: STEP 3
                
                # STEP 4: Extending with the end character
                else:
                    l_plus = l + c
                    if len(l) > 0 and c == l[-1]:
                        Pnb[t][l_plus] += ctc[t][c_ix] * Pb[t - 1][l]
                        Pnb[t][l] += ctc[t][c_ix] * Pnb[t - 1][l]
                # END: STEP 4

                    # STEP 5: Extending with any other non-blank character and LM constraints
                    elif len(l.replace(' ', '')) > 0 and c in (' ', end_char):
                        lm_prob = lm(l_plus.strip(' '+end_char)) ** alpha
                        Pnb[t][l_plus] += lm_prob * ctc[t][c_ix] * (Pb[t - 1][l] + Pnb[t - 1][l])
                    else:
                        Pnb[t][l_plus] += ctc[t][c_ix] * (Pb[t - 1][l] + Pnb[t - 1][l])
                    # END: STEP 5

                    # STEP 6: Make use of discarded prefixes
                    if l_plus not in A_prev:
                        Pb[t][l_plus] += ctc[t][blank_index] * (Pb[t - 1][l_plus] + Pnb[t - 1][l_plus])
                        Pnb[t][l_plus] += ctc[t][c_ix] * Pnb[t - 1][l_plus]
                    # END: STEP 6

        # STEP 7: Select most probable prefixes
        A_next = Pb[t] + Pnb[t]
        sorter = lambda l: A_next[l] * (len(W(l)) + 1) ** beta
        A_prev = sorted(A_next, key=sorter, reverse=True)[:k]
        # END: STEP 7
    if len(A_prev) ==0:
        A_prev=['']
    if return_weights:
        return A_prev[0],A_next[A_prev[0]] * (len(W(A_prev[0])) + 1) ** beta
    return A_prev[0]
    #For N-best decode, return A_prev[0:N] - not tested yet.

class PrefixBeamSearchLMDecoder(Decoder):
    def __init__(self,lm_path,labels,blank_index=0,k=5,alpha=0.3,beta=5,prune=1e-3):
        """
        Args:
            lm_path (str): The path to the kenlm language model.
            labels (list(str)): A list of the characters.
            blank_index (int): The index of the blank character in the `labels` parameter.
            k (int): The beam width. Will keep the 'k' most likely candidates at each timestep.
            alpha (float): The language model weight. Should usually be between 0 and 1.
            beta (float): The language model compensation term. The higher the 'alpha', the higher the 'beta'.
            prune (float): Only extend prefixes with chars with an emission probability higher than 'prune'.
        """
        super(PrefixBeamSearchLMDecoder, self).__init__(labels,blank_index)
        if lm_path:
            import kenlm
            self.lm = kenlm.Model(lm_path)
            self.lm_weigh = lambda f: 10**(self.lm.score(f))
        else:
            self.lm_weigh = lambda s: 1
        self.k =k
        self.alpha=alpha
        self.beta=beta
        self.prune=prune
        
    def decode(self,probs,sizes=None):
        if len(probs.shape) == 2: # Single    
            return prefix_beam_search(probs,self.labels,self.blank_index,self.lm_weigh,self.k,self.alpha,self.beta,self.prune)
        elif len(probs.shape) == 3: # Batch
            return [self.decode(prob) for prob in probs]
        else:
            raise RuntimeError('Decoding with wrong shape: %s, expected either [Batch X Frames X Labels] or [Frames X Labels]' % str(probs.shape))
    
