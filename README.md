# Wav2Letter_pytorch

Implementation of Wav2Letter using PyTorch.
Creates a network based on the [Wav2Letter](https://arxiv.org/abs/1609.03193) architecture, trained with CTC loss.

## Features

* Minimalist code, designed to be a white box - dive into the code!
* Train End-To-End ASR models, including Wav2Letter and Jasper.
* Uses [Hydra](https://hydra.cc/docs/intro) for easy configuration and usage
* Uses [PyTorch Lightning](https://www.pytorchlightning.ai/) for simplify training
* Beam search decoding integrated with kenlm language models.

# Installation

Install Python 3.6 or higher. 

Clone the repository (or download it) and install according to the requirements file.
```
pip install -r requirements.txt
```

# Usage

## LibriSpeech Example
Run ```examples/librispeech.sh``` to download and prepare the data, and start training with a single script.

You can use the ```data/prepare_librispeech.py``` script to prepare other subsets of the Librispeech dataset. 
Run it with ```--help``` for more information.



## Training
Most simple example:
```
python train.py data.train_manifest TRAIN.csv data.val_manifest VALID.csv
```
Training a Jasper model is as simple as: ```python train.py model.name=jasper ...```

To train with multiple GPU's, mixed precision, or many other options, see the [Pytorch-Lightning Trainer](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-class-api) documentation. 

Many elements of the model and training are managed via configuration files and command line flags via Hydra.  
This includes the audio preprocessing configuration, the optimizer and learning-rate scheduler, and the number and configuration of layers. See the configuration directory for more details.  
To see the entire configuration, run ```python train.py [optional overrides] --cfg=job```

## Testing/Inference
WIP!
To evaluate a trained model on a test set (has to be in the same format as the training set):

```
python test.py --model-path models/wav2Letter.pth --test-manifest /path/to/test_manifest.csv --cuda
```

To see the decoded outputs compared to the test data, run with either ```--print-samples``` or ```print-all```.

You can use a LM during decoding. The LM is expected to be a valid ARPA model, loaded with kenlm. Add ```--lm-path``` to use it. See ```--beam-search-params``` to fine tune your parameters for beam search.

### Custom Datasets
To create a custom dataset, create a Pandas Dataframe with the columns ```audio_filepath, text``` and save it using ``` df.to_csv(path) ```.
Alternatively, you can create a .json file - each line contains a json of a sample with at least ```audio_filepath, text``` as fields.
You can add reading specific sections of audio files by adding ```offset``` and ```duration``` fields (in seconds). The values 0 and -1 are the default values, respectively, and cause reading the entire audio file.
If you use a sample rate other than 16K, specify it using ```model.audio_conf.sample_rate=8000``` for example. 

### Different languages
In addition to English, Hebrew is supported.

To use, run with ```--labels hebrew```. Note that some terminals and consoles do not display UTF-8 properly.


## Differences from article

There are some subtle differences between this implementation and the original.  
We use CTC loss instead of ASG, which leads to a small difference in labels definition and output interpretation.  
We currently use spectrogram features instead of MFCC, which achieved the best results in the original article.  
Some of the network hyperparameters are different - convolution kernel sizes, strides, and default sample rate.  

## Acknowledgements
This work was originally based off [Silversparro's Wav2Letter](https://github.com/silversparro/wav2letter.pytorch).
That work was inspired by  the [deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch) repository of [Sean Naren](https://github.com/SeanNaren). 
The prefix-beam-search algorithm is based on [corticph](https://github.com/corticph/prefix-beam-search) with minor edits.
