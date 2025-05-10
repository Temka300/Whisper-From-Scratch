# Whisper-From-Scratch

A reimplementation of OpenAI's Whisper model for speech recognition, built from scratch using PyTorch and trained on the Common Voice dataset. This project focuses on Mongolian language support but can be adapted for other languages.

## Features

- From-scratch implementation of Whisper architecture
- Support for training on low-memory GPUs (6GB+)
- Customizable tokenizer for different languages
- Evaluation using WER (Word Error Rate) and CER (Character Error Rate)
- Transcription support for single files, directories, and evaluation against reference transcripts

## Requirements

- Python 3.7+
- PyTorch 1.7+
- CUDA-capable GPU (for efficient training)
- Common Voice dataset (or other ASR datasets)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Whisper-From-Scratch.git
cd Whisper-From-Scratch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

## Data Preparation

This code is designed to work with the Common Voice dataset. Download the dataset for your target language from [Common Voice](https://commonvoice.mozilla.org/datasets).

Extract the dataset to a directory, for example: `cv-corpus-16.1`

## Training

### Standard Training

To train the model with default settings:

```bash
./run.sh
```

This script:
1. Installs dependencies
2. Trains a small Whisper model on the Mongolian Common Voice dataset
3. Saves the model checkpoints to `whisper-mongolian-from-scratch` directory

### Low-Memory Training

For GPUs with limited memory (e.g., 6GB):

```bash
./run_6gb.sh
```

This script uses a smaller batch size and the "tiny" Whisper model configuration.

### Very Low-Memory Training

For even more aggressive memory optimization:

```bash
./run_6gb_lowmem.sh
```

This script enables additional memory optimizations:
- Automatic Mixed Precision (AMP)
- Custom memory management
- Smaller batch sizes and model configurations

### Custom Training

You can also run the training script directly with custom parameters:

```bash
python train.py \
    --local-data-dir "path/to/dataset" \
    --whisper-size "tiny" \
    --keep-chars "your-character-set" \
    --train-batch-size 8 \
    --eval-batch-size 4 \
    --num-epochs 10 \
    --learning-rate 1e-5 \
    --output-dir "output-folder"
```

Key parameters:
- `whisper-size`: Model size (`tiny` or `small`)
- `keep-chars`: Character set for the target language
- `train-batch-size`: Number of examples per batch (reduce for less memory usage)
- `output-dir`: Directory to save model checkpoints and logs

## Transcription

After training, you can transcribe audio files using:

```bash
python transcribe.py --model_path "whisper-mongolian-from-scratch/best_model.pt" --mode single --audio_path "path/to/audio.wav"
```

### Transcription Modes

- `single`: Transcribe a single audio file
- `directory`: Transcribe all audio files in a directory
- `batch`: Process multiple files with optimized batching
- `evaluate`: Calculate WER against reference transcriptions

Examples:

```bash
# Transcribe all files in a directory
python transcribe.py --model_path "whisper-mongolian-from-scratch/best_model.pt" --mode directory --audio_dir "path/to/audio/folder" --output_file "transcriptions.txt"

# Evaluate against reference transcriptions
python transcribe.py --model_path "whisper-mongolian-from-scratch/best_model.pt" --mode evaluate --audio_dir "path/to/audio/folder" --reference_file "references.txt"
```

## Model Architecture

This implementation follows the original Whisper architecture:
- Audio encoder: Processes mel spectrograms through convolutional and transformer layers
- Text decoder: Generates text transcriptions using cross-attention with the encoder outputs

Available model sizes:
- `tiny`: 80M parameters
- `small`: 244M parameters

## Customizing for Other Languages

To adapt this code for other languages:
1. Update the `--keep-chars` parameter to include the characters in your target language
2. Prepare a dataset for your language (Common Voice or other ASR corpus)
3. Train using one of the provided scripts

## Troubleshooting

### Memory Issues

If you encounter CUDA out-of-memory errors:
- Use a smaller model size (`tiny` instead of `small`)
- Reduce batch size
- Use the low-memory training script (`run_6gb_lowmem.sh`)
- Set environment variables for PyTorch memory management:
  ```
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
  ```

### Training Speed

- For faster training, ensure you have a recent CUDA version and compatible PyTorch build
- Consider using Automatic Mixed Precision (enabled in the low-memory script)

## License

This project is open-source and available under [LICENSE].
