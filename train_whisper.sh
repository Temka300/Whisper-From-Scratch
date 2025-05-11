#!/bin/bash

# Script to train Whisper model with fixes for validation/inference discrepancy

# Default parameters
OUTPUT_DIR="whisper-model-new"
DATA_DIR="cv-corpus-16.1"
WHISPER_SIZE="tiny"
NUM_EPOCHS=1
LEARNING_RATE=5e-5
TEMPERATURE=1.0
TRAIN_BATCH_SIZE=1
EVAL_BATCH_SIZE=1
SAVE_EVERY=50
EVAL_STEPS=20
LOGGING_STEPS=5
MAX_SAMPLES=30
RESUME_FROM=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --whisper-size)
      WHISPER_SIZE="$2"
      shift 2
      ;;
    --num-epochs)
      NUM_EPOCHS="$2"
      shift 2
      ;;
    --learning-rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --train-batch-size)
      TRAIN_BATCH_SIZE="$2"
      shift 2
      ;;
    --eval-batch-size)
      EVAL_BATCH_SIZE="$2"
      shift 2
      ;;
    --save-every)
      SAVE_EVERY="$2"
      shift 2
      ;;
    --eval-steps)
      EVAL_STEPS="$2"
      shift 2
      ;;
    --logging-steps)
      LOGGING_STEPS="$2"
      shift 2
      ;;
    --max-samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --resume-from)
      RESUME_FROM="$2"
      shift 2
      ;;
    --help)
      echo "Usage: ./train_whisper.sh [options]"
      echo "Options:"
      echo "  --output-dir DIR         Directory to save models (default: $OUTPUT_DIR)"
      echo "  --data-dir DIR           Directory containing the dataset (default: $DATA_DIR)"
      echo "  --whisper-size SIZE      Model size to use (default: $WHISPER_SIZE, choices: tiny, small)"
      echo "  --num-epochs NUM         Number of training epochs (default: $NUM_EPOCHS)"
      echo "  --learning-rate RATE     Learning rate (default: $LEARNING_RATE)"
      echo "  --temperature TEMP       Temperature for validation (default: $TEMPERATURE)"
      echo "  --train-batch-size SIZE  Training batch size (default: $TRAIN_BATCH_SIZE)"
      echo "  --eval-batch-size SIZE   Evaluation batch size (default: $EVAL_BATCH_SIZE)"
      echo "  --save-every NUM         Save model every N steps (default: $SAVE_EVERY)"
      echo "  --eval-steps NUM         Evaluate every N steps (default: $EVAL_STEPS)"
      echo "  --logging-steps NUM      Log metrics every N steps (default: $LOGGING_STEPS)"
      echo "  --max-samples NUM        Maximum number of samples to use (default: $MAX_SAMPLES)"
      echo "  --resume-from PATH       Resume training from checkpoint (default: none)"
      echo "  --help                   Show this help message and exit"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Print training parameters
echo "Training Whisper model with the following parameters:"
echo "  Output directory: $OUTPUT_DIR"
echo "  Data directory: $DATA_DIR"
echo "  Model size: $WHISPER_SIZE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Temperature: $TEMPERATURE"
echo "  Training batch size: $TRAIN_BATCH_SIZE"
echo "  Evaluation batch size: $EVAL_BATCH_SIZE"
echo "  Save every: $SAVE_EVERY steps"
echo "  Evaluate every: $EVAL_STEPS steps"
echo "  Log every: $LOGGING_STEPS steps"
echo "  Max samples: $MAX_SAMPLES"
if [ -n "$RESUME_FROM" ]; then
  echo "  Resume from: $RESUME_FROM"
fi

# Run the training script with lower memory limits
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
echo "Setting CUDA memory allocator to limit fragment size (max_split_size_mb:128)"

# Run the training script
ARGS="--output-dir $OUTPUT_DIR --data-dir $DATA_DIR --whisper-size $WHISPER_SIZE --num-epochs $NUM_EPOCHS --learning-rate $LEARNING_RATE --temperature $TEMPERATURE --train-batch-size $TRAIN_BATCH_SIZE --eval-batch-size $EVAL_BATCH_SIZE --save-every $SAVE_EVERY --eval-steps $EVAL_STEPS --logging-steps $LOGGING_STEPS --max-samples $MAX_SAMPLES"

if [ -n "$RESUME_FROM" ]; then
  ARGS="$ARGS --resume-from $RESUME_FROM"
fi

python train_with_fixes.py $ARGS

echo "Training completed!" 