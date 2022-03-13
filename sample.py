from wav2letter import Conv1dBlock
import data.data_loader
import data.label_sets
from collections import namedtuple, OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


class Wav2LetterPretrained(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        conv_blocks = []
        layer =  Conv1dBlock(64,256,(11,),2,0.2)
        conv_blocks.append(('conv1d_0',layer))
        layer =  Conv1dBlock(256,256,(11,),1,0.2)
        conv_blocks.append(('conv1d_1',layer))
        layer =  Conv1dBlock(256,29,(1,),1,0.2,bn=False,activation_use=False)
        conv_blocks.append(('conv1d_2',layer))
        self.conv1ds = nn.Sequential(OrderedDict(conv_blocks))
        self.load_state_dict(state_dict)
        self.eval()

    def forward(self,x):
        ''' Run the model on samples. Input: [Batch, Channels, Length]
        Output: [Batch, Length, Characters'''

        scores = self.conv1ds(x).transpose(1,2)
        probs = F.log_softmax(scores,dim=-1)
        return probs

checkpoint = torch.load('/home/assafmushkin/workshop/wav2letter_pytorch/lightning_logs/version_18/checkpoints/epoch=299-step=7499.ckpt')
model = Wav2LetterPretrained(checkpoint['state_dict'])

print(model)

audio_config_template = namedtuple('AudioConfig','sample_rate window_size window_stride window')
audio_conf = audio_config_template(sample_rate=16000, window_size=0.02, window_stride=0.01, window='hamming')
spect_extractor = data.data_loader.SpectrogramExtractor(audio_conf)


sample_path = "./extracted/LibriSpeech/dev-clean/7850/281318/7850-281318-0015.flac"
audio = data.data_loader.load_audio(sample_path)
spect = spect_extractor.extract(audio)
print(f'input shape:{spect.shape}')
outputs = model(spect.unsqueeze(0))
print(f'output shape:{outputs.shape}')
english_labels = data.label_sets.labels_map['english']
print(f'English labels: {english_labels}')
