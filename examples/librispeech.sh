#!/usr/bin/env bash
python check_requirements.py
python ../data/prepare_librispeech.py --help # train set
python ../data/prepare_librispeech.py --help # test set
echo ${DATA_DIR}/herro
# python ../train.py data.train_manifest=train_clean.json data.val_manifest=/ACLP-Nimble/Users/Assaf/LibriSpeech/debug_clean.json trainer.default_root_dir=.
