import argparse
import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
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

# Enable memory savings
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def get_audio_features(batch):
    """Extract log-mel spectrogram features with explicit memory management."""
    audio = batch["audio"]["array"]
    # Convert to float32 before processing
    audio = np.array(audio, dtype=np.float32)
    audio = pad_or_trim(audio.flatten())
    mel = log_mel_spectrogram(torch.from_numpy(audio))
    
    # Debug shape
    print(f"Initial mel shape: {mel.shape}")
    
    # Free memory
    del audio
    gc.collect()
    
    # Store as numpy array - we store it with (features, time) for easier processing later
    # The model expects [batch, features, time] but we handle that in the process_batch function
    return {"input_features": mel.squeeze(0).cpu().numpy(), "text": batch["text"]}

class CustomDataset(Dataset):
    def __init__(self, features, texts):
        self.features = features
        self.texts = texts
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return {
            "input_features": self.features[idx],
            "text": self.texts[idx]
        }

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
    
    def process_batch(batch, device):
        # Free memory first
        torch.cuda.empty_cache()
        gc.collect()
        
        # Move data to device
        # The model expects audio_features to be of shape [batch_size, n_mels, seq_len]
        # but our data is [batch_size, seq_len, n_mels]
        audio_features = torch.stack([torch.from_numpy(b["input_features"]) for b in batch]).to(device, dtype=torch.float32)
        print(f"Audio features shape before: {audio_features.shape}")

        # Transpose to [batch, 80, seq_len] if needed
        if audio_features.shape[1] != 80:
            audio_features = audio_features.transpose(1, 2)
        print(f"Audio features shape after transpose: {audio_features.shape}")

        # Ensure the context length is correct
        max_len = 1500
        if audio_features.shape[2] > max_len:
            audio_features = audio_features[:, :, :max_len]
        print(f"Final audio features shape: {audio_features.shape}")
        
        texts = [b["text"] for b in batch]
        token_ids, attention_mask = tokenizer.batch_encode(texts)
        
        # Truncate to max decoder context length
        max_text_ctx = 448
        if token_ids.shape[1] > max_text_ctx:
            token_ids = token_ids[:, :max_text_ctx]
            attention_mask = attention_mask[:, :max_text_ctx]
            
        token_ids = token_ids.to(device)
        attention_mask = attention_mask.to(device)
        return audio_features, token_ids, attention_mask
    
    def evaluate_model(model, dataset, num_batches=None):
        model.eval()
        all_preds = []
        all_refs = []

        # Process max 100 examples for evaluation to save memory
        eval_subset = dataset.select(range(min(100, len(dataset))))
        
        # Process in smaller chunks
        processed_features = []
        processed_texts = []
        
        # Process data in small batches
        chunk_size = 10
        for i in range(0, len(eval_subset), chunk_size):
            chunk = eval_subset.select(range(i, min(i + chunk_size, len(eval_subset))))
            
            # Process features for this chunk
            processed_chunk = [get_audio_features(example) for example in chunk]
            
            for item in processed_chunk:
                processed_features.append(item["input_features"])
                processed_texts.append(item["text"])
            
            # Force cleanup
            torch.cuda.empty_cache()
            gc.collect()
        
        # Create custom dataset
        custom_dataset = CustomDataset(processed_features, processed_texts)
        
        dataloader = DataLoader(
            custom_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda x: x  # Return list of dictionaries as is
        )
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
                if num_batches and i >= num_batches:
                    break
                
                audio_features, token_ids, attention_mask = process_batch(batch, device)
                
                # Forward pass with half precision
                with torch.autocast(device_type=device, dtype=torch.float16):
                    logits = model(audio_features, token_ids[:, :-1], attention_mask[:, :-1])
                
                pred_ids = torch.argmax(logits, dim=-1)
                predictions = tokenizer.batch_decode(pred_ids)
                
                all_preds.extend(predictions)
                all_refs.extend([b["text"] for b in batch])
                
                # Clear memory after each batch
                del audio_features, token_ids, attention_mask, logits, pred_ids
                torch.cuda.empty_cache()
                gc.collect()
        
        wer = wer_metric.compute(predictions=all_preds, references=all_refs)
        cer = cer_metric.compute(predictions=all_preds, references=all_refs)
        
        return {"wer": wer, "cer": cer}
    
    # Training loop
    global_step = 0
    best_wer = float("inf")
    
    # Process data in chunks to save memory
    chunk_size = 100  # Process 100 examples at a time
    
    print("Starting training...")
    optimizer.zero_grad()
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0
        
        # Process the dataset in chunks
        for chunk_start in range(0, len(train_dataset), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(train_dataset))
            print(f"Processing chunk {chunk_start}-{chunk_end} of {len(train_dataset)}")
            
            # Get a subset of the dataset
            train_subset = train_dataset.select(range(chunk_start, chunk_end))
            
            # Process features for this chunk
            processed_features = []
            processed_texts = []
            
            for example in tqdm(train_subset, desc="Processing audio"):
                processed = get_audio_features(example)
                processed_features.append(processed["input_features"])
                processed_texts.append(processed["text"])
            
            # Create custom dataset
            custom_dataset = CustomDataset(processed_features, processed_texts)
            
            train_dataloader = DataLoader(
                custom_dataset,
                batch_size=args.train_batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=lambda x: x  # Return list of dictionaries as is
            )
            
            # Train on this chunk
            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}, Chunk {chunk_start}-{chunk_end}")):
                audio_features, token_ids, attention_mask = process_batch(batch, device)
                
                # Forward pass with half precision
                with torch.autocast(device_type=device, dtype=torch.float16):
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
                
                # Clear memory after each batch
                del audio_features, token_ids, attention_mask, logits
                torch.cuda.empty_cache()
                gc.collect()
                
                epoch_loss += loss.item() * gradient_accumulation_steps
        
            # Clear memory after each chunk
            del processed_features, processed_texts, custom_dataset, train_dataloader
            torch.cuda.empty_cache()
            gc.collect()
        
        epoch_loss /= (len(train_dataset) // args.train_batch_size)
        print(f"Epoch {epoch+1} average loss: {epoch_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-data-dir', default="cv-corpus-16.1")
    parser.add_argument('--train-datasets', default=None)
    parser.add_argument('--eval-datasets', default=None)
    parser.add_argument('--whisper-size', default='tiny')
    parser.add_argument('--keep-chars', default=KEEP_CHARS)
    parser.add_argument('--train-batch-size', default=2, type=int)
    parser.add_argument('--eval-batch-size', default=1, type=int)
    parser.add_argument('--num-epochs', default=10, type=int)
    parser.add_argument('--learning-rate', default=1e-5, type=float)
    parser.add_argument('--logging-steps', default=100, type=int)
    parser.add_argument('--eval-steps', default=1000, type=int)
    parser.add_argument('--eval-batches', default=10, type=int)
    parser.add_argument('--output-dir', default='whisper-mongolian-from-scratch')
    
    args = parser.parse_args()
    show_argparse(args)
    train(args) 