# -*- coding: utf-8 -*-
from wav2letter import Wav2Letter
import torch
import numpy as np

def test_sanity():
    model = Wav2Letter(input_size=1,mid_layers=1)
    
def test_packaging():
    model = Wav2Letter(input_size=1)
    package = model.serialize(model)
    new_model = Wav2Letter.load_model_package(package)
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in new_model.parameters())
    
def test_output_dimensions():
    input_size = 64
    labels = list(range(24))
    model = Wav2Letter(labels,input_size=input_size)    
    batch_size = 11
    max_length_of_audio = 79
    inp = torch.ones(batch_size,input_size,max_length_of_audio)
    input_lengths = torch.randint(max_length_of_audio,(batch_size,))
    import math
    max_output_length = math.floor(max_length_of_audio/model.get_scaling_factor())
    model.eval()
    out = model(inp,input_lengths)
    assert out.shape ==  torch.Size((batch_size,max_output_length,len(labels)))
    assert np.allclose(np.ones((batch_size,max_output_length)),out.detach().numpy().sum(axis=2))
    model.train()
    out = model(inp,input_lengths)
    assert np.allclose(np.ones((batch_size,max_output_length)),np.exp(out.detach().numpy()).sum(axis=2))
    