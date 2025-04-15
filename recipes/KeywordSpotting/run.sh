gpu_ids="0"

# Calculate the number of processes based on the number of GPUs
if [[ -z "$num_processes" ]]; then
  num_processes=$(echo "$gpu_ids" | tr "," "\n" | wc -l)
fi

# Separate datasets and neurons
dataset="gsc_v2_command"
neuron="clif"

echo "Running on bmi-5 [Training]"

torchrun_bin="/home/zysong/miniconda3/envs/audiozen/bin/torchrun"

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="${gpu_ids}" "${torchrun_bin}" \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes=1 \
    --nproc-per-node="$num_processes" \
    run.py \
    --config_path "conf/${dataset}/${neuron}.yaml" \
    --do_train true \
    --do_eval false \
    --do_predict false \
    --output_dir "exp/${dataset}/${neuron}"

# for i in "${!datasets[@]}"; do
#   dataset="${datasets[$i]}"
#   neuron="${neurons[$i]}"
#   config_name="${neuron}_${dataset}"
  
#   echo "Running with config: ${config_name}"
  
#   OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="${gpu_ids}" "${torchrun_bin}" \
#       --rdzv-backend=c10d \
#       --rdzv-endpoint=localhost:0 \
#       --nnodes=1 \
#       --nproc-per-node="$num_processes" \
#       run.py \
#       --config_path "conf/${config_name}.yaml" \
#       --do_train true \
#       --do_eval false \
#       --do_predict false \
#       --output_dir "exp/${dataset}/${neuron}"

# done
