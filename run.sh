# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run training with local dataset
python train.py \
    --local-data-dir "cv-corpus-16.1" \
    --whisper-size "small" \
    --keep-chars " абвгдеёжзийклмноөпрстуүфхцчшъыьэюя.,?!" \
    --train-batch-size 16 \
    --eval-batch-size 8 \
    --num-epochs 10 \
    --learning-rate 1e-5 \
    --logging-steps 100 \
    --eval-steps 1000 \
    --output-dir "whisper-mongolian-from-scratch"
