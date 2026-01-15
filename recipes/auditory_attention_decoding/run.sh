#!/bin/bash

gpu_ids="2"

# Calculate the number of processes based on the number of GPUs
if [[ -z "$num_processes" ]]; then
  num_processes=$(echo "$gpu_ids" | tr "," "\n" | wc -l)
fi

# Separate datasets and neurons
dataset="DTU"
neuron="lif"

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="${gpu_ids}" torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes=1 \
    --nproc-per-node="$num_processes" \
    run.py \
    --config_path "conf/${dataset}/${neuron}.yaml" \
    --do_train true \
    --do_eval false \
    --do_predict false \
    --output_dir "exp"

