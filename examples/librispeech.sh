#!/usr/bin/env bash
echo "This script downloads the LibriSpeech dev-clean and test-clean subsets, and trains a model for 1 epoch." 
echo "Usage: bash librispeech.sh [wav2letter_pytorch directory - optional]"
base_dir=${1:-$(dirname $(dirname "$(readlink -f "$0")"))}
python ${base_dir}/examples/check_requirements.py
if [ $? -ne 0 ]; then
	echo "Requirements not found!"
	exit
fi
python ${base_dir}/data/prepare_librispeech.py --subset dev-clean --manifest_path dev_clean.csv
python ${base_dir}/data/prepare_librispeech.py --subset test-clean --manifest_path test_clean.csv
if [ $? -ne 0 ]; then
	echo "Data preparation failed"
	exit
fi
python ${base_dir}/train.py data.train_manifest=dev_clean.csv data.val_manifest=test_clean.csv trainer.max_epochs=1
if [ $? -ne 0 ]; then
	echo "Training failed"
	exit
fi
echo Training finished successfully!
echo Tensorboard logs were saved to directory "./lightning_logs". Call 'tensorboard --logdir ./lightning_logs' to view the logs generated during training.
