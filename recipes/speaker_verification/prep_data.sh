#!/usr/bin/env bash
set -e

VOX_ROOT=/datasets/Voxceleb/vox1
VOX_DEV_PATH=${VOX_ROOT}/dev
VOX_veri_PATH=${VOX_ROOT}/wav
META_DIR=metadata

echo "==> Build train csv of vox1 dev"
python3 scripts/build_datalist.py \
    --extension wav \
    --dataset_dir ${VOX_ROOT}/dev \
    --data_list_path ${META_DIR}/train_vox1.csv

echo "==> Build test pairs txt"
# Build the 
python3 scripts/build_datalist.py \
    --extension wav \
    --veri_test_path ${META_DIR}/veri_set.txt \
    --vox1_test_path ${META_DIR}/vox1_test.txt \
    --wav_root ${VOX_veri_PATH}

echo "Done."
