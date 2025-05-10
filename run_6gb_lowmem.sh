#!/bin/bash

# Set CUDA memory management for low memory devices
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Use this for enabling AMP (Automatic Mixed Precision)
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Run training with local dataset
python train_low_memory.py \
    --local-data-dir "cv-corpus-16.1" \
    --whisper-size "tiny" \
    --keep-chars " абвгдеёжзийклмноөпрстуүфхцчшъыьэюя.,?!" \
    --train-batch-size 1 \
    --eval-batch-size 1 \
    --num-epochs 5 \
    --learning-rate 1e-5 \
    --logging-steps 50 \
    --eval-steps 500 \
    --eval-batches 5 \
    --output-dir "whisper-mongolian-from-scratch" 