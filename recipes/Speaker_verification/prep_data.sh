#Build train set on voxceleb1
python3 scripts/build_datalist.py \
        --extension wav \
        --dataset_dir /datasets/Voxceleb/vox1/dev \
        --data_list_path metadata/train_vox1.csv