#!/usr/bin/env bash
echo "This script downloads the LibriSpeech \"dev clean\" dataset and prepares a CSV file ready for training." 
echo "Usage: bash run_example.sh [wav2letter_pytorch directory - optional]"
base_dir=${1:-$(dirname $(dirname "$(readlink -f "$0")"))}
python ${base_dir}/examples/check_requirements.py
if [ $? -ne 0 ]; then
	echo "Requirements not found!"
	exit
fi
python ${base_dir}/data/prepare_librispeech.py --subset dev-clean --manifest_path dev_clean.csv
head -n 101 dev_clean.csv > mini_100.csv
if [ $? -ne 0 ]; then
	echo "Data preparation failed"
	exit
fi
echo "Running sanity check"
python ${base_dir}/train.py data.train_manifest=mini_100.csv data.val_manifest=mini_100.csv trainer.max_epochs=1
if [ $? -ne 0 ]; then
	echo "Training failed"
	exit
fi
echo All checks done! You're ready for the workshop

