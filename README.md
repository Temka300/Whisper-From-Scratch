# Whisper-From-Scratch
Using Common Voice mn 16.1 dataset 

`whisper_train.py`

Usage:
```bash
python whisper_train.py \
  --data-dir cv-corpus-16.1 \
  --whisper-size tiny \
  --num-epochs 5 \
  --learning-rate 1e-4 \
  --train-batch-size 1 \
  --eval-batch-size 1
```
