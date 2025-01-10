#!/bin/bash

## ==================
## Define the GPU IDs
## ==================
gpu_ids="0"
# gpu_ids="4,5,6,7"
# gpu_ids="0,1,2,3,4,5,6,7"

# Calculate the number of processes based on the number of GPUs
if [[ -z "$num_processes" ]]; then
  num_processes=$(echo "$gpu_ids" | tr "," "\n" | wc -l)
fi

default_config_name="ALIF"
exp_name="ALIF_baseline"

echo "Running on bmi-5 [Training]"
torchrun_bin="/home/zysong/miniconda3/envs/audiozen/bin/torchrun"

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="${gpu_ids}" "${torchrun_bin}" \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes=1 \
    --nproc-per-node="$num_processes" \
    run.py \
    --config_path "conf/${default_config_name}.yaml" \
    --do_eval false \
    --output_dir "exp/${exp_name}"