#!/bin/bash
export ASCEND_RT_VISIBLE_DEVICES=4,5

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${WORLD_SIZE:-2}

entry_file="example_inference.py"

model_id="/mnt/disk3/dxw/Qwen-Image-Edit"
input_image="/home/dxw/qwenimage/modular/cute_cat.png"

msrun --worker_num=${NPROC_PER_NODE} \
    --local_worker_num=${NPROC_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --log_dir="logs/infer" \
    --join=True \
    ${entry_file} \
    --model_id "${model_id}" \
    --input_image "${input_image}"
