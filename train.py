import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
import evaluate
from tqdm import tqdm
import numpy as np

from multiple_datasets.utils import show_argparse
from multiple_datasets.dataset_utils import (
    prepare_datasets,
    KEEP_CHARS
)
from whisper_from_scratch.model import Whisper
from whisper_from_scratch.tokenizer import WhisperTokenizer
from whisper_from_scratch.feature_extraction import log_mel_spectrogram, pad_or_trim

def get_audio_features(batch):
    """Extract log-mel spectrogram features."""
    audio = batch["audio"]["array"]
    # Convert to float32 before processing
    audio = np.array(audio, dtype=np.float32)
    audio = pad_or_trim(audio.flatten())
    mel = log_mel_spectrogram(torch.from_numpy(audio))
    return {"input_features": mel, "text": batch["text"]}

def train(args):
    # Initialize model and tokenizer
    print(f"Initializing Whisper {args.whisper_size} model...")
    dims = Whisper.get_default_dims(args.whisper_size)
    model = Whisper(dims)
    tokenizer = WhisperTokenizer()
    
    # Load datasets with memory efficient settings
    print("Loading datasets...")
    train_dataset, eval_dataset = prepare_datasets(
        args.train_datasets,
        args.eval_datasets,
        args.keep_chars,
        args.local_data_dir
    )
    
    if train_dataset is None:
        raise ValueError("Failed to load training dataset")
    
    # Setup training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    writer = SummaryWriter(args.output_dir)
    
    # Memory efficiency: Enable gradient accumulation
    gradient_accumulation_steps = max(1, 32 // args.train_batch_size)
    effective_batch_size = args.train_batch_size * gradient_accumulation_steps
    print(f"Using gradient accumulation with {gradient_accumulation_steps} steps")
    print(f"Effective batch size: {effective_batch_size}")
    
    # Metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    def process_batch(batch):
        audio_features = torch.stack([torch.from_numpy(b["input_features"]) for b in batch]).to(device)
        texts = [b["text"] for b in batch]
        token_ids, attention_mask = tokenizer.batch_encode(texts)
        token_ids = token_ids.to(device)
        attention_mask = attention_mask.to(device)
        return audio_features, token_ids, attention_mask
    
    def evaluate_model(model, dataset, num_batches=None):
        model.eval()
        all_preds = []
        all_refs = []
        
        dataloader = DataLoader(
            dataset.map(get_audio_features),
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=list,
            num_workers=0
        )
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
                if num_batches and i >= num_batches:
                    break
                
                audio_features, token_ids, attention_mask = process_batch(batch)
                logits = model(audio_features, token_ids[:, :-1], attention_mask[:, :-1])
                pred_ids = torch.argmax(logits, dim=-1)
                predictions = tokenizer.batch_decode(pred_ids)
                
                all_preds.extend(predictions)
                all_refs.extend([b["text"] for b in batch])
                
                if device == "cuda":
                    torch.cuda.empty_cache()
        
        wer = wer_metric.compute(predictions=all_preds, references=all_refs)
        cer = cer_metric.compute(predictions=all_preds, references=all_refs)
        
        return {"wer": wer, "cer": cer}
    
    # Training loop
    global_step = 0
    best_wer = float("inf")
    
    train_dataloader = DataLoader(
        train_dataset.map(
            get_audio_features,
            remove_columns=train_dataset.column_names
        ),
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=list,
        num_workers=0,
        pin_memory=False
    )
    
    print("Starting training...")
    optimizer.zero_grad()
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            audio_features, token_ids, attention_mask = process_batch(batch)
            
            # Forward pass
            logits = model(audio_features, token_ids[:, :-1], attention_mask[:, :-1])
            
            # Calculate loss and scale by accumulation steps
            loss = criterion(
                logits.view(-1, dims["n_vocab"]),
                token_ids[:, 1:].contiguous().view(-1)
            ) / gradient_accumulation_steps
            
            loss.backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                if global_step % args.logging_steps == 0:
                    writer.add_scalar("Loss/train", loss.item() * gradient_accumulation_steps, global_step)
                
                if global_step % args.eval_steps == 0:
                    metrics = evaluate_model(model, eval_dataset, num_batches=args.eval_batches)
                    
                    writer.add_scalar("WER", metrics["wer"], global_step)
                    writer.add_scalar("CER", metrics["cer"], global_step)
                    
                    print(f"Step {global_step}: WER = {metrics['wer']:.4f}, CER = {metrics['cer']:.4f}")
                    
                    if metrics["wer"] < best_wer:
                        best_wer = metrics["wer"]
                        save_path = os.path.join(args.output_dir, "best_model.pt")
                        model.save(save_path)
                        print(f"New best model saved with WER: {best_wer:.4f}")
                
                if device == "cuda" and global_step % 10 == 0:
                    torch.cuda.empty_cache()
            
            epoch_loss += loss.item() * gradient_accumulation_steps
        
        epoch_loss /= len(train_dataloader)
        print(f"Epoch {epoch+1} average loss: {epoch_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-data-dir', default="cv-corpus-16.1")
    parser.add_argument('--train-datasets', default=None)
    parser.add_argument('--eval-datasets', default=None)
    parser.add_argument('--whisper-size', default='small')
    parser.add_argument('--keep-chars', default=KEEP_CHARS)
    parser.add_argument('--train-batch-size', default=16, type=int)
    parser.add_argument('--eval-batch-size', default=8, type=int)
    parser.add_argument('--num-epochs', default=10, type=int)
    parser.add_argument('--learning-rate', default=1e-5, type=float)
    parser.add_argument('--logging-steps', default=100, type=int)
    parser.add_argument('--eval-steps', default=1000, type=int)
    parser.add_argument('--eval-batches', default=None, type=int)
    parser.add_argument('--output-dir', default='whisper-mongolian')
    
    args = parser.parse_args()
    show_argparse(args)
    train(args)
