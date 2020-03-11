# wav2Letter.pytorch

Implementation of Wav2Letter using PyTorch.
Creates a network based on the [Wav2Letter](https://arxiv.org/abs/1609.03193) architecture, trained with the CTC activation function.

## Features

* Train Wav2Letter.
* Language model support using kenlm.
* Easy start/stop capabilities in the event of crash or hard stop during training.
* Tensorboard support for visualizing training graphs.

# Installation

Several libraries are needed to be installed for training to work. I will assume that everything is being installed in
an Anaconda installation on Ubuntu.

Install [PyTorch](https://github.com/pytorch/pytorch#installation) if you haven't already.

Install pytorch audio (if on windows, see note):
```
sudo apt-get install sox libsox-dev libsox-fmt-all
git clone https://github.com/pytorch/audio.git
cd audio
pip install cffi
python setup.py install
```

Finally clone this repo and run this within the repo:
```
pip install -r requirements.txt
```
## Windows installation

As of March 11, the repo runs correctly on a Windows 10 machine. We expect to keep Windows support for the forseeable future.

However, Windows is recommended only for spot checks - for actual training, use Linux.

We recommend installing PyTorch with an Anaconda installation, and Microsoft Visual C++ Build Tools for kenlm and python-levenshtein.

Tensorboard has not been tested on windows yet, so we recommend running with --no-tensorboard.

As of March 11, torchaudio does not support Windows, however it is [in progress](https://github.com/pytorch/audio/issues/425)

# Usage

### Custom Dataset

To create a custom dataset you must create a CSV file containing the locations of the training data. This has to be in the format of:

```
/path/to/audio.wav,transcription
/path/to/audio2.wav,transcription
...
```

The first path is to the audio file, and the second is the text containing the transcript on one line. This can then be used as stated below.

## Training

```
python train.py --train-manifest data/train_manifest.csv --val-manifest data/val_manifest.csv
```

Use `python train.py --help` for more parameters and options.

There is [Tensorboard](https://github.com/lanpa/tensorboard-pytorch) support to visualize training. Follow the instructions to set up. To use:

```
python train.py --tensorboard --logdir log_dir/ # Make sure the Tensorboard instance is made pointing to this log directory
```

## Testing/Inference

To evaluate a trained model on a test set (has to be in the same format as the training set):

```
python test.py --model-path models/wav2Letter.pth --test-manifest /path/to/test_manifest.csv --cuda
```

## Acknowledgements
This work was originally based off [Silversparro's Wav2Letter](https://github.com/silversparro/wav2letter.pytorch).
That work was inspired by  the [deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch) repository of [Sean Naren](https://github.com/SeanNaren). 
The prefix-beam-search algorithm is based on [corticph](https://github.com/corticph/prefix-beam-search) with minor edits.