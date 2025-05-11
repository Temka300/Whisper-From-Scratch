#!/usr/bin/env python3
"""
whisper_train.py - Standalone script for training a Whisper model
Usage: python whisper_train.py --whisper-size small --num-epochs 1
"""
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
from multiple_datasets.dataset_utils import prepare_datasets, KEEP_CHARS
from whisper_from_scratch.model import Whisper
from whisper_from_scratch.tokenizer import WhisperTokenizer
from whisper_from_scratch.feature_extraction import log_mel_spectrogram, pad_or_trim

# Configure for memory efficiency
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

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

def get_audio_features(batch):
    """Extract log-mel spectrogram features with explicit memory management."""
    audio = batch["audio"]["array"]
    # Convert to float32 before processing
    audio = np.array(audio, dtype=np.float32)
    audio = pad_or_trim(audio.flatten())
    mel = log_mel_spectrogram(torch.from_numpy(audio))
    
    # Free memory
    del audio
    gc.collect()
    
    # Store as numpy array
    return {"input_features": mel.squeeze(0).cpu().numpy(), "text": batch["text"]}

def process_batch(batch, device, dims):
    # Free memory first
    torch.cuda.empty_cache()
    gc.collect()
    
    # Move data to device
    audio_features = torch.stack([torch.from_numpy(b["input_features"]) for b in batch]).to(device, dtype=torch.float32)
    
    # Transpose to [batch, 80, seq_len] if needed
    if audio_features.shape[1] != dims["n_mels"]:
        audio_features = audio_features.transpose(1, 2)
    
    # Ensure the context length is correct
    max_len = dims["n_audio_ctx"]
    if audio_features.shape[2] > max_len:
        audio_features = audio_features[:, :, :max_len]
    
    texts = [b["text"] for b in batch]
    token_ids, attention_mask = WhisperTokenizer().batch_encode(texts)
    
    # Truncate to max decoder context length
    max_text_ctx = dims["n_text_ctx"]
    if token_ids.shape[1] > max_text_ctx:
        token_ids = token_ids[:, :max_text_ctx]
        attention_mask = attention_mask[:, :max_text_ctx]
        
    token_ids = token_ids.to(device)
    attention_mask = attention_mask.to(device)
    return audio_features, token_ids, attention_mask

def evaluate_model(model, dataset, tokenizer, dims, device="cuda", num_batches=None):
    model.eval()
    all_preds = []
    all_refs = []
    
    # Metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

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
        batch_size=1,  # Evaluate one at a time
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: x  # Return list of dictionaries as is
    )
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if num_batches and i >= num_batches:
                break
            
            audio_features, token_ids, attention_mask = process_batch(batch, device, dims)
            
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
    
    # Calculate metrics
    wer = wer_metric.compute(predictions=all_preds, references=all_refs)
    cer = cer_metric.compute(predictions=all_preds, references=all_refs)
    
    # Print some examples
    for i in range(min(3, len(all_preds))):
        print(f"Example {i}:")
        print(f"  Prediction: '{all_preds[i]}'")
        print(f"  Reference:  '{all_refs[i]}'")
    
    return {"wer": wer, "cer": cer}

def train_whisper(
    data_dir="cv-corpus-16.1",
    model_size="small",
    output_dir="whisper-model",
    num_epochs=1,
    learning_rate=1e-4,
    save_every=50,
    train_batch_size=1,
    eval_batch_size=1,
    eval_steps=100,
    logging_steps=10,
    resume_from=None
):
    """Train a Whisper model from scratch or continue training"""
    print(f"Training Whisper {model_size} model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize model and tokenizer
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        # First load checkpoint to get dimensions
        checkpoint = torch.load(resume_from)
        saved_dims = None
        actual_tensor_dims = None
        
        # Check first if we can determine dimensions directly from tensor shapes
        if "model_state_dict" in checkpoint:
            for key, tensor in checkpoint["model_state_dict"].items():
                if 'encoder.positional_embedding' in key:
                    embed_size = tensor.shape[1]
                    print(f"Detected embedding size from weights: {embed_size}")
                    if embed_size == 384:
                        actual_tensor_dims = Whisper.get_default_dims("tiny")
                        print(f"Using 'tiny' model architecture based on actual weights")
                    elif embed_size == 768:
                        actual_tensor_dims = Whisper.get_default_dims("small")
                        print(f"Using 'small' model architecture based on actual weights")
                    elif embed_size == 1280:
                        actual_tensor_dims = Whisper.get_default_dims("medium")
                        print(f"Using 'medium' model architecture based on actual weights")
                    break
        
        # Check if dims were saved directly
        if "dims" in checkpoint:
            saved_dims = checkpoint["dims"]
            print(f"Dimensions from checkpoint metadata: {saved_dims}")
            
            # Compare with actual tensor dimensions if available
            if actual_tensor_dims and saved_dims["n_audio_state"] != actual_tensor_dims["n_audio_state"]:
                print(f"WARNING: Mismatch between saved dimensions and actual tensor shapes!")
                print(f"Saved metadata says {saved_dims['n_audio_state']} but weights are {actual_tensor_dims['n_audio_state']}")
                print(f"Using actual tensor dimensions for model creation")
                saved_dims = actual_tensor_dims
        
        # If no saved dims but we detected from tensors
        if saved_dims is None and actual_tensor_dims is not None:
            saved_dims = actual_tensor_dims
            
        # Create model with correct dimensions
        if saved_dims:
            print(f"Creating model with dimensions: {saved_dims}")
            dims = saved_dims  # Store the dimensions in the dims variable
            model = Whisper(dims)
        else:
            # Fallback if dimensions couldn't be determined
            print(f"Warning: Could not determine dimensions from checkpoint. Using requested size: {model_size}")
            dims = Whisper.get_default_dims(model_size)
            model = Whisper(dims)
            
        # Now load state dict
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("Creating new model")
        dims = Whisper.get_default_dims(model_size)
        model = Whisper(dims)
        
    model = model.to(device)
    tokenizer = WhisperTokenizer()
    
    # Load datasets
    print(f"Loading datasets from {data_dir}...")
    train_dataset, eval_dataset = prepare_datasets(
        train_datasets=None,
        eval_datasets=None,
        keep_chars=KEEP_CHARS,
        local_data_dir=data_dir
    )
    
    if train_dataset is None:
        raise ValueError("Failed to load training dataset")
    
    if eval_dataset is None:
        print("No eval dataset provided, using a subset of training data")
        eval_dataset = train_dataset.select(range(min(100, len(train_dataset))))
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup training
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    writer = SummaryWriter(output_dir)
    
    # Memory efficiency: Enable gradient accumulation
    gradient_accumulation_steps = max(1, 32 // train_batch_size)
    effective_batch_size = train_batch_size * gradient_accumulation_steps
    print(f"Using gradient accumulation with {gradient_accumulation_steps} steps")
    print(f"Effective batch size: {effective_batch_size}")
    
    # Process data in chunks to save memory
    chunk_size = 100  # Process 100 examples at a time
    
    # Training loop
    global_step = 0
    best_wer = float('inf')
    
    print(f"Starting training for {num_epochs} epochs, saving every {save_every} steps")
    print(f"Evaluating every {eval_steps} steps")
    
    optimizer.zero_grad()
    
    for epoch in range(num_epochs):
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
                batch_size=train_batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=lambda x: x  # Return list of dictionaries as is
            )
            
            # Train on this chunk
            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}, Chunk {chunk_start}-{chunk_end}")):
                audio_features, token_ids, attention_mask = process_batch(batch, device, dims)
                
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
                    
                    # Log loss
                    if global_step % logging_steps == 0:
                        writer.add_scalar("Loss/train", loss.item() * gradient_accumulation_steps, global_step)
                        print(f"\nStep {global_step}, Loss: {loss.item() * gradient_accumulation_steps:.4f}")
                        print(f"Sample: '{processed_texts[0][:50]}...'")
                    
                    # Evaluate model
                    if global_step % eval_steps == 0:
                        print("\nEvaluating model...")
                        metrics = evaluate_model(model, eval_dataset, tokenizer, dims, device)
                        
                        writer.add_scalar("WER", metrics["wer"], global_step)
                        writer.add_scalar("CER", metrics["cer"], global_step)
                        
                        print(f"Step {global_step}: WER = {metrics['wer']:.4f}, CER = {metrics['cer']:.4f}")
                        
                        # Save best model
                        if metrics["wer"] < best_wer:
                            best_wer = metrics["wer"]
                            save_path = os.path.join(output_dir, "best_model.pt")
                            model.save(save_path)
                            print(f"New best model saved with WER: {best_wer:.4f}")
                    
                    # Save checkpoint
                    if global_step % save_every == 0:
                        save_path = os.path.join(output_dir, f"model_step_{global_step}.pt")
                        model.save(save_path)
                        print(f"Saved model checkpoint: {save_path}")
                
                # Clear memory after each batch
                del audio_features, token_ids, attention_mask, logits
                torch.cuda.empty_cache()
                gc.collect()
                
                epoch_loss += loss.item() * gradient_accumulation_steps
        
            # Clear memory after each chunk
            del processed_features, processed_texts, custom_dataset, train_dataloader
            torch.cuda.empty_cache()
            gc.collect()
        
        # End of epoch
        avg_loss = epoch_loss / (len(train_dataset) // train_batch_size)
        print(f"Epoch {epoch+1} completed with average loss: {avg_loss:.4f}")
        
        # Save epoch checkpoint
        save_path = os.path.join(output_dir, f"model_epoch_{epoch+1}.pt")
        model.save(save_path)
        print(f"Saved model at epoch {epoch+1}: {save_path}")
    
    writer.close()
    print("Training completed!")
    print(f"Final model saved at: {os.path.join(output_dir, f'model_epoch_{num_epochs}.pt')}")
    print(f"Best model saved at: {os.path.join(output_dir, 'best_model.pt')}")
    return os.path.join(output_dir, 'best_model.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Whisper model")
    parser.add_argument('--data-dir', default="cv-corpus-16.1", help="Directory containing dataset")
    parser.add_argument('--whisper-size', default='small', choices=['tiny', 'small'], help="Model size")
    parser.add_argument('--output-dir', default='whisper-model', help="Directory to save models")
    parser.add_argument('--num-epochs', default=1, type=int, help="Number of training epochs")
    parser.add_argument('--learning-rate', default=1e-4, type=float, help="Learning rate")
    parser.add_argument('--train-batch-size', default=1, type=int, help="Training batch size")
    parser.add_argument('--eval-batch-size', default=1, type=int, help="Evaluation batch size")
    parser.add_argument('--save-every', default=50, type=int, help="Save model every N steps")
    parser.add_argument('--eval-steps', default=100, type=int, help="Evaluate every N steps")
    parser.add_argument('--logging-steps', default=10, type=int, help="Log metrics every N steps")
    parser.add_argument('--resume-from', default=None, help="Resume training from checkpoint")
    
    args = parser.parse_args()
    show_argparse(args)
    
    train_whisper(
        data_dir=args.data_dir,
        model_size=args.whisper_size,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        save_every=args.save_every,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        resume_from=args.resume_from
    ) 