#!/bin/bash

# Add CUDA memory management flags to reduce memory usage
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Run training with local dataset
python train.py \
    --local-data-dir "cv-corpus-16.1" \
    --whisper-size "tiny" \
    --keep-chars " абвгдеёжзийклмноөпрстуүфхцчшъыьэюя.,?!" \
    --train-batch-size 2 \
    --eval-batch-size 1 \
    --num-epochs 10 \
    --learning-rate 1e-5 \
    --logging-steps 100 \
    --eval-steps 1000 \
    --eval-batches 10 \
    --output-dir "whisper-mongolian-from-scratch" 