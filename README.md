# Wav2Letter_pytorch

Implementation of Wav2Letter using PyTorch.
Creates a network based on the [Wav2Letter](https://arxiv.org/abs/1609.03193) architecture, trained with the CTC activation function.

## Features

* Minimalist code, designed to be a white box.
* Train End-To-End ASR models, including Wav2Letter and Jasper.
* Beam search decoding integrated with kenlm language models.
* Uses [Hydra](https://hydra.cc/docs/intro) for easy configuration and usage
* Uses [PyTorch Lightning](https://www.pytorchlightning.ai/) for simplify training

# Installation

Install Python 3.6 or higher. 

Clone the repository (or download it) and install according to the requirements file.
```
pip install -r requirements.txt
```
## Windows installation

As of December 20, 2020, Windows is supported but recommended only for spot checks - for actual training, use Linux.

We recommend installing PyTorch with an Anaconda installation, and Microsoft Visual C++ Build Tools for kenlm and python-levenshtein.

FFmpeg for windows is a portable binary [from here](https://www.ffmpeg.org/download.html#build-windows)

# Usage

### LibriSpeech Dataset
Run ```examples/librispeech.sh``` to download and prepare the data, and start training with a single script.

You can use the ```data/prepare_librispeech.py``` script to prepare other subsets of the Librispeech dataset. 
Run it with ```--help``` for more information.

### Custom Datasets
To create a custom dataset, create a CSV file containing audio location and text pairs.
This can be in the following format:

```
/path/to/audio.wav,transcription
/path/to/audio2.wav,transcription
...
```

Alternatively, create a Pandas Dataframe with the columns ```filepath, text``` and save it using ``` df.to_csv(path) ```.

If you use a sample rate other than 8K, specify it using ```model.audio_conf.sample_rate=16000``` for example. 

### Different languages
In addition to English, Hebrew is supported.

To use, run with ```--labels hebrew```. Note that some terminals and consoles do not display UTF-8 properly.


## Training

```
python train.py --train-manifest data/train_manifest.csv --val-manifest data/val_manifest.csv
```

Use `python train.py --help` for more parameters and options.

There is [Tensorboard](https://github.com/lanpa/tensorboard-pytorch) support to visualize training. Follow the instructions to set up. To use:

```
python train.py --tensorboard --logdir log_dir/ # Make sure the Tensorboard instance is made pointing to this log directory
```
### Continue training existing model
To continue training from an existing model, run with ```--continue-from MODEL_PATH```.

Note that ```--layers, --labels, --window_size, --window_stride, --window, --sample_rate``` are all determined from the configuration of the loaded model, and are ignored. 

## Testing/Inference

To evaluate a trained model on a test set (has to be in the same format as the training set):

```
python test.py --model-path models/wav2Letter.pth --test-manifest /path/to/test_manifest.csv --cuda
```

To see the decoded outputs compared to the test data, run with either ```--print-samples``` or ```print-all```.

You can use a LM during decoding. The LM is expected to be a valid ARPA model, loaded with kenlm. Add ```--lm-path``` to use it. See ```--beam-search-params``` to fine tune your parameters for beam search.

## Differences from article

There are some subtle differences between this implementation and the original.

We use CTC loss instead of ASG, which leads to a small difference in labels definition and output interpretation.

We currently use spectrogram features instead of MFCC, which achieved the best results in the original article.

Some of the network hyperparameters are different - convolution kernel sizes, strides, and default sample rate.

## Acknowledgements
This work was originally based off [Silversparro's Wav2Letter](https://github.com/silversparro/wav2letter.pytorch).
That work was inspired by  the [deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch) repository of [Sean Naren](https://github.com/SeanNaren). 
The prefix-beam-search algorithm is based on [corticph](https://github.com/corticph/prefix-beam-search) with minor edits.