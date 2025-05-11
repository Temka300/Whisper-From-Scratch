#!/usr/bin/env python3
"""
train_with_fixes.py - Script for training a Whisper model with fixes for validation/inference discrepancy
Usage: python train_with_fixes.py --whisper-size tiny --num-epochs 1
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
import logging

from src.multiple_datasets.utils import show_argparse
from src.multiple_datasets.dataset_utils import prepare_datasets, KEEP_CHARS
from src.whisper_from_scratch.model import Whisper
from src.whisper_from_scratch.tokenizer import WhisperTokenizer
from src.whisper_from_scratch.feature_extraction import log_mel_spectrogram, pad_or_trim

# Configure for memory efficiency
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    """Process a batch for model input with dimension validation."""
    # Free memory first
    if torch.cuda.is_available():
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
    elif audio_features.shape[2] < max_len:
        # Pad if too short
        padding = torch.zeros((audio_features.shape[0], audio_features.shape[1], max_len - audio_features.shape[2]), 
                             device=device, dtype=audio_features.dtype)
        audio_features = torch.cat([audio_features, padding], dim=2)
    
    # Get tokenized text
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

def evaluate_model_inference_style(model, dataset, tokenizer, dims, temperature=1.0, device="cuda", num_batches=None):
    """
    Evaluate model using the same decoding strategy as in inference.
    This is the key fix for validation/inference discrepancy.
    """
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
        if torch.cuda.is_available():
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
            
            # Get encoder output - this is key for matching inference
            encoder_output = model.encoder(audio_features)
            
            # Initialize decoder input with start token - inference style
            start_token_id = 50258  # Whisper start token
            current_ids = torch.ones((audio_features.shape[0], 1), dtype=torch.long, device=device) * start_token_id
            current_mask = torch.ones_like(current_ids, device=device)
            
            # Generate text autoregressively - exactly like inference
            max_length = 100
            end_token_id = 50257  # Whisper end token
            generated_ids = []
            
            for j in range(max_length):
                # Forward pass using cached encoder output
                logits = model.decoder(current_ids, encoder_output, current_mask)
                
                # Apply temperature and select next token
                next_token_logits = logits[:, -1, :] / temperature
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Store generated token
                generated_ids.append(next_token)
                
                # Append to sequence
                current_ids = torch.cat([current_ids, next_token], dim=1)
                current_mask = torch.cat([current_mask, torch.ones_like(next_token, device=device)], dim=1)
                
                # Break if end token is generated
                if (next_token == end_token_id).all():
                    break
            
            # Combine all generated tokens
            pred_ids = torch.cat(generated_ids, dim=1)
            predictions = tokenizer.batch_decode(pred_ids)
            
            all_preds.extend(predictions)
            all_refs.extend([b["text"] for b in batch])
            
            # Clear memory after each batch
            del audio_features, token_ids, attention_mask, encoder_output, current_ids, current_mask, logits, pred_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    # Calculate metrics
    wer = wer_metric.compute(predictions=all_preds, references=all_refs)
    cer = cer_metric.compute(predictions=all_preds, references=all_refs)
    
    # Print some examples
    for i in range(min(3, len(all_preds))):
        logger.info(f"Example {i}:")
        logger.info(f"  Prediction: '{all_preds[i]}'")
        logger.info(f"  Reference:  '{all_refs[i]}'")
    
    return {"wer": wer, "cer": cer}

def train_whisper_with_fixes(
    data_dir="cv-corpus-16.1",
    model_size="tiny",
    output_dir="whisper-model-new",
    num_epochs=2,
    learning_rate=5e-5,
    temperature=1.0,
    save_every=50,
    train_batch_size=1,
    eval_batch_size=1,
    eval_steps=20,
    logging_steps=5,
    max_samples=50,
    resume_from=None
):
    """
    Train a Whisper model with fixes for validation/inference discrepancy.
    Key improvements:
    1. Uses the same decoding strategy in validation as in inference
    2. Applies temperature during validation
    3. Ensures correct dimension handling for audio features
    4. Explicit encoder-decoder separation during training
    """
    logger.info(f"Training Whisper {model_size} model with validation/inference fixes...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Initialize model and tokenizer
    if resume_from and os.path.exists(resume_from):
        logger.info(f"Resuming from checkpoint: {resume_from}")
        try:
            # Load checkpoint
            checkpoint = torch.load(resume_from, map_location=device)
            
            # Determine dimensions
            if "dims" in checkpoint:
                dims = checkpoint["dims"]
                logger.info(f"Using dimensions from checkpoint: {dims}")
            else:
                dims = Whisper.get_default_dims(model_size)
                logger.info(f"Using default dimensions for {model_size}")
            
            # Create model
            model = Whisper(dims)
            
            # Load state dict based on checkpoint format
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            
            logger.info(f"Model loaded successfully from {resume_from}")
        except Exception as e:
            logger.warning(f"Could not load model from {resume_from}: {e}")
            dims = Whisper.get_default_dims(model_size)
            model = Whisper(dims)
            logger.info(f"Created new model with size: {model_size}")
    else:
        logger.info(f"Creating new model with size: {model_size}")
        dims = Whisper.get_default_dims(model_size)
        model = Whisper(dims)
    
    model = model.to(device)
    tokenizer = WhisperTokenizer()
    
    # Target audio context length from model dimensions
    audio_ctx_length = dims["n_audio_ctx"]
    logger.info(f"Model expects audio context length: {audio_ctx_length}")
    
    # Load datasets
    logger.info(f"Loading datasets from {data_dir}...")
    train_dataset, eval_dataset = prepare_datasets(
        train_datasets=None,
        eval_datasets=None,
        keep_chars=KEEP_CHARS,
        local_data_dir=data_dir
    )
    
    if train_dataset is None:
        raise ValueError("Failed to load training dataset")
    
    # Limit dataset size if requested
    if max_samples and len(train_dataset) > max_samples:
        logger.info(f"Limiting training dataset to {max_samples} samples (from {len(train_dataset)})")
        train_dataset = train_dataset.select(range(max_samples))
    
    # Create validation set if none exists
    if eval_dataset is None:
        logger.info("No validation dataset found. Creating validation set from training data.")
        # Use part of the training data as validation
        val_size = min(max_samples // 5, len(train_dataset) // 5) if max_samples else len(train_dataset) // 5
        
        # Create validation set from the end of the training set
        val_indices = list(range(len(train_dataset) - val_size, len(train_dataset)))
        eval_dataset = train_dataset.select(val_indices)
        
        # Adjust training set to remove validation samples
        train_dataset = train_dataset.select(list(range(len(train_dataset) - val_size)))
        
        logger.info(f"Created validation dataset with {len(eval_dataset)} samples")
        logger.info(f"Adjusted training dataset to {len(train_dataset)} samples")
    elif max_samples and len(eval_dataset) > max_samples // 5:
        val_limit = max_samples // 5
        logger.info(f"Limiting validation dataset to {val_limit} samples (from {len(eval_dataset)})")
        eval_dataset = eval_dataset.select(range(val_limit))
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup training
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    writer = SummaryWriter(output_dir)
    
    # Memory efficiency settings
    gradient_accumulation_steps = max(1, 16 // train_batch_size)
    effective_batch_size = train_batch_size * gradient_accumulation_steps
    logger.info(f"Using gradient accumulation with {gradient_accumulation_steps} steps")
    logger.info(f"Effective batch size: {effective_batch_size}")
    
    # Enable automatic mixed precision for memory efficiency if available
    if torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        logger.info("Using mixed precision training for memory efficiency")
    else:
        scaler = None
        logger.info("Mixed precision not available, using full precision")
    
    # Process data in smaller chunks to save memory - reduce chunk size
    chunk_size = 10 if max_samples <= 50 else 20
    
    # Training loop
    global_step = 0
    best_wer = float('inf')
    
    logger.info(f"Starting training for {num_epochs} epochs, saving every {save_every} steps")
    logger.info(f"Evaluating every {eval_steps} steps with temperature = {temperature}")
    
    optimizer.zero_grad()
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        # Process the dataset in chunks
        for chunk_start in range(0, len(train_dataset), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(train_dataset))
            logger.info(f"Processing chunk {chunk_start}-{chunk_end} of {len(train_dataset)}")
            
            # Get a subset of the dataset
            train_subset = train_dataset.select(range(chunk_start, chunk_end))
            
            # Process features for this chunk - smaller batches to avoid memory spikes
            processed_features = []
            processed_texts = []
            
            # Process 5 examples at a time to avoid memory spikes
            mini_batch_size = 5
            for mini_batch_start in range(0, len(train_subset), mini_batch_size):
                mini_batch_end = min(mini_batch_start + mini_batch_size, len(train_subset))
                
                for example_idx in tqdm(range(mini_batch_start, mini_batch_end), desc="Processing audio"):
                    example = train_subset[example_idx]
                    processed = get_audio_features(example)
                    processed_features.append(processed["input_features"])
                    processed_texts.append(processed["text"])
                    
                    # Force garbage collection after each example
                    gc.collect()
                
                # Force cleanup between mini-batches
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            
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
                # Clear memory before processing
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                audio_features, token_ids, attention_mask = process_batch(batch, device, dims)
                
                # Use automatic mixed precision for forward pass if available
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        # Forward pass - explicitly separate encoder and decoder
                        encoder_output = model.encoder(audio_features)
                        logits = model.decoder(token_ids[:, :-1], encoder_output, attention_mask[:, :-1])
                        
                        # Calculate loss and scale by accumulation steps
                        raw_loss = criterion(
                            logits.reshape(-1, dims["n_vocab"]),
                            token_ids[:, 1:].reshape(-1)
                        )
                        loss = raw_loss / gradient_accumulation_steps
                    
                    # Scale gradients and backward pass
                    scaler.scale(loss).backward()
                else:
                    # Forward pass - explicitly separate encoder and decoder
                    encoder_output = model.encoder(audio_features)
                    logits = model.decoder(token_ids[:, :-1], encoder_output, attention_mask[:, :-1])
                    
                    # Calculate loss and scale by accumulation steps
                    raw_loss = criterion(
                        logits.reshape(-1, dims["n_vocab"]),
                        token_ids[:, 1:].reshape(-1)
                    )
                    loss = raw_loss / gradient_accumulation_steps
                    
                    # Log raw loss for every batch regardless of accumulation
                    logger.info(f"Batch loss: {raw_loss.item():.4f} (accumulated: {loss.item():.4f})")
                    
                    loss.backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Update weights with gradient scaling if available
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    optimizer.zero_grad()
                    global_step += 1
                    
                    # Log loss
                    if global_step % logging_steps == 0:
                        writer.add_scalar("Loss/train", loss.item() * gradient_accumulation_steps, global_step)
                        logger.info(f"Step {global_step}, Loss: {loss.item() * gradient_accumulation_steps:.4f}")
                        logger.info(f"Sample: '{processed_texts[0][:50]}...'")
                    
                    # Evaluate model using inference-like decoding
                    if global_step % eval_steps == 0 or batch_idx == len(train_dataloader) - 1:  # Also evaluate at end of each chunk
                        logger.info("Evaluating model with inference-style decoding...")
                        
                        # Quick sample test - generate from the current batch to verify decoding
                        with torch.no_grad():
                            # Get sample audio
                            sample_audio = audio_features[:1]  # Just use the first example in batch
                            
                            # Get encoder output
                            sample_encoder_output = model.encoder(sample_audio)
                            
                            # Generate text - same as inference
                            start_token_id = 50258
                            current_ids = torch.ones((1, 1), dtype=torch.long, device=device) * start_token_id
                            current_mask = torch.ones_like(current_ids, device=device)
                            
                            # Generate tokens
                            generated_text = ""
                            max_gen_length = 50
                            end_token_id = 50257
                            
                            for j in range(max_gen_length):
                                # Forward pass using cached encoder output
                                with torch.cuda.amp.autocast(enabled=scaler is not None):
                                    logits = model.decoder(current_ids, sample_encoder_output, current_mask)
                                
                                # Apply temperature and select next token
                                next_token_logits = logits[:, -1, :] / temperature
                                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                                
                                # Append to sequence
                                current_ids = torch.cat([current_ids, next_token], dim=1)
                                current_mask = torch.cat([current_mask, torch.ones_like(next_token, device=device)], dim=1)
                                
                                # Break if end token is generated
                                if next_token.item() == end_token_id:
                                    break
                            
                            # Decode generated tokens
                            generated_ids = current_ids[:, 1:]  # Remove start token
                            sample_prediction = tokenizer.batch_decode(generated_ids)[0]
                            sample_reference = processed_texts[0]
                            
                            logger.info(f"SAMPLE GENERATION TEST:")
                            logger.info(f"  Prediction: '{sample_prediction}'")
                            logger.info(f"  Reference:  '{sample_reference}'")
                        
                        # Run full evaluation on subset
                        try:
                            metrics = evaluate_model_inference_style(
                                model, eval_dataset, tokenizer, dims, 
                                temperature=temperature, device=device, 
                                num_batches=5  # Limit evaluation to 5 batches to save memory
                            )
                            
                            writer.add_scalar("WER", metrics["wer"], global_step)
                            writer.add_scalar("CER", metrics["cer"], global_step)
                            
                            logger.info(f"Step {global_step}: WER = {metrics['wer']:.4f}, CER = {metrics['cer']:.4f}")
                            
                            # Save best model
                            if metrics["wer"] < best_wer:
                                best_wer = metrics["wer"]
                                save_path = os.path.join(output_dir, "best_model.pt")
                                torch.save({
                                    "model_state_dict": model.state_dict(),
                                    "dims": dims,
                                    "model_size": model_size
                                }, save_path)
                                logger.info(f"New best model saved with WER: {best_wer:.4f}")
                                
                        except Exception as e:
                            logger.error(f"Error during evaluation: {e}")
                            logger.info("Continuing training despite evaluation error")
                    
                    # Save checkpoint
                    if global_step % save_every == 0:
                        save_path = os.path.join(output_dir, f"model_step_{global_step}.pt")
                        torch.save({
                            "model_state_dict": model.state_dict(),
                            "dims": dims,
                            "model_size": model_size
                        }, save_path)
                        logger.info(f"Saved model checkpoint: {save_path}")
                
                # Clear memory after each batch
                del audio_features, token_ids, attention_mask, encoder_output, logits
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                epoch_loss += loss.item() * gradient_accumulation_steps
        
            # Clear memory after each chunk
            del processed_features, processed_texts, custom_dataset, train_dataloader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # End of epoch
        avg_loss = epoch_loss / (len(train_dataset) // train_batch_size)
        logger.info(f"Epoch {epoch+1} completed with average loss: {avg_loss:.4f}")
        
        # Save epoch checkpoint
        save_path = os.path.join(output_dir, f"model_epoch_{epoch+1}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "dims": dims,
            "model_size": model_size
        }, save_path)
        logger.info(f"Saved model at epoch {epoch+1}: {save_path}")
    
    writer.close()
    logger.info("Training completed!")
    logger.info(f"Final model saved at: {os.path.join(output_dir, f'model_epoch_{num_epochs}.pt')}")
    logger.info(f"Best model saved at: {os.path.join(output_dir, 'best_model.pt')}")
    return os.path.join(output_dir, 'best_model.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Whisper model with fixes for validation/inference discrepancy")
    parser.add_argument('--data-dir', default="cv-corpus-16.1", help="Directory containing dataset")
    parser.add_argument('--whisper-size', default='tiny', choices=['tiny', 'small'], help="Model size")
    parser.add_argument('--output-dir', default='whisper-model-new', help="Directory to save models")
    parser.add_argument('--num-epochs', default=2, type=int, help="Number of training epochs")
    parser.add_argument('--learning-rate', default=5e-5, type=float, help="Learning rate")
    parser.add_argument('--temperature', default=1.0, type=float, help="Temperature for validation")
    parser.add_argument('--train-batch-size', default=1, type=int, help="Training batch size")
    parser.add_argument('--eval-batch-size', default=1, type=int, help="Evaluation batch size")
    parser.add_argument('--save-every', default=50, type=int, help="Save model every N steps")
    parser.add_argument('--eval-steps', default=20, type=int, help="Evaluate every N steps")
    parser.add_argument('--logging-steps', default=5, type=int, help="Log metrics every N steps")
    parser.add_argument('--max-samples', default=50, type=int, help="Maximum number of samples to use")
    parser.add_argument('--resume-from', default=None, help="Resume training from checkpoint")
    
    args = parser.parse_args()
    
    if "show_argparse" in globals():
        show_argparse(args)
    else:
        logger.info(f"Arguments: {args}")
    
    train_whisper_with_fixes(
        data_dir=args.data_dir,
        model_size=args.whisper_size,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        save_every=args.save_every,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        max_samples=args.max_samples,
        resume_from=args.resume_from
    ) 