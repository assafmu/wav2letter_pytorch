#!/usr/bin/env bash
echo "This script downloads dev-clean and test-clean, and trains a model for 1 epoch." 
base_dir=/home/assafm/Git/wav2letter_pytorch/
python ${base_dir}/examples/check_requirements.py
python ${base_dir}/data/prepare_librispeech.py --subset dev-clean --manifest_path dev_clean.csv
python ${base_dir}/data/prepare_librispeech.py --subset test-clean --manifest_path test_clean.csv
python ${base_dir}/train.py data.train_manifest=dev_clean.csv data.val_manifest=test_clean.csv trainer.default_root_dir=. trainer.max_epochs=1
echo Tensorboard logs were saved to X. Call 'tensorboard --logdir X' to view the logs generated during training.