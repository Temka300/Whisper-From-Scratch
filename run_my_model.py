#!/usr/bin/env python3
"""
run_my_model.py - Simple script to run the trained Whisper model for transcription
Usage: python run_my_model.py --audio-path PATH_TO_AUDIO
"""

import argparse
import torch
import torchaudio
import os
import numpy as np

from whisper_from_scratch.model import Whisper
from whisper_from_scratch.tokenizer import WhisperTokenizer
from whisper_from_scratch.feature_extraction import log_mel_spectrogram, pad_or_trim

def transcribe(audio_path, model_path, temperature=0.0, use_greedy=True):
    """Transcribe an audio file using the specified model"""
    print(f"Loading model from {model_path}")
    model = Whisper.load(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    tokenizer = WhisperTokenizer()
    
    print(f"Transcribing: {audio_path}")
    
    # Load and process audio
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Resample if needed
    if sample_rate != 16000:
        print(f"Resampling from {sample_rate}Hz to 16000Hz")
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, 
            new_freq=16000
        )(waveform)
    
    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Make sure it's the right shape
    waveform = waveform.squeeze(0).numpy()
    
    # Pad or trim to standard length
    waveform = pad_or_trim(waveform)
    
    # Extract log mel spectrogram features
    mel = log_mel_spectrogram(torch.from_numpy(waveform)).to(device)
    
    # Prepare feature shape for model - ensure it's [batch_size, n_mels, time]
    if mel.dim() == 2:  # If shape is [80, time]
        audio_features = mel.unsqueeze(0)  # Make it [1, 80, time]
    elif mel.dim() == 3 and mel.shape[0] == 1:  # If shape is [1, 80, time]
        audio_features = mel
    else:
        # Force reshape if needed
        print(f"Unexpected mel shape: {mel.shape}, reshaping")
        audio_features = mel.view(1, 80, -1)
    
    print(f"Audio features shape: {audio_features.shape}")
    
    # Ensure dimensions don't exceed model's capacity
    if audio_features.shape[2] > 1500:
        print(f"Truncating audio features from length {audio_features.shape[2]} to 1500")
        audio_features = audio_features[:, :, :1500]
    
    # Generate transcription
    model.eval()
    transcription = ""
    with torch.no_grad():
        if use_greedy:
            # Similar to validation approach - greedy decoding
            print("Using greedy decoding approach")
            # Start with BOS token
            bos_token_id = tokenizer.bos_token_id
            input_ids = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            
            # Get encoder output (just once)
            encoder_output = model.encoder(audio_features)
            print(f"Encoder output shape: {encoder_output.shape}")
            
            # Maximum output length
            max_length = 448
            current_ids = input_ids
            current_mask = attention_mask
            
            for _ in range(max_length - 1):
                # Forward pass
                with torch.autocast('cuda', dtype=torch.float16):
                    logits = model(audio_features, current_ids, current_mask)
                
                # Get next token (greedy)
                next_token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                
                # Append to sequence
                current_ids = torch.cat([current_ids, next_token_id], dim=1)
                current_mask = torch.cat([current_mask, torch.ones_like(next_token_id)], dim=1)
                
                # Stop if we generated EOS
                if next_token_id.item() == tokenizer.eos_token_id:
                    break
            
            # Decode the sequence
            transcription = tokenizer.decode(current_ids[0].tolist())
        else:
            # Use nucleus sampling with temperature
            print(f"Using sampling with temperature={temperature}")
            # Start with BOS token
            bos_token_id = tokenizer.bos_token_id
            current_ids = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)
            current_mask = torch.ones_like(current_ids, dtype=torch.long)
            
            # Maximum output length
            max_length = 448
            
            for _ in range(max_length - 1):
                # Forward pass
                with torch.autocast('cuda', dtype=torch.float16):
                    outputs = model(audio_features, current_ids, current_mask)
                
                # Get logits for the next token and apply temperature
                next_token_logits = outputs[:, -1, :]
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                # Sample from the distribution or use argmax (greedy) if temp = 0
                if temperature == 0:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                else:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                current_ids = torch.cat([current_ids, next_token], dim=1)
                current_mask = torch.cat([current_mask, torch.ones_like(next_token)], dim=1)
                
                # Stop if we generated EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break
            
            # Decode the sequence
            transcription = tokenizer.decode(current_ids[0].tolist())
    
    return transcription

def main():
    parser = argparse.ArgumentParser(description="Run trained Whisper model")
    parser.add_argument('--audio-path', required=True, help="Path to audio file for transcription")
    parser.add_argument('--model-path', default="/home/temka/code-play/github_repo/whisper-multiple-hf-datasets/whisper-model-new/best_model.pt", 
                       help="Path to the trained model")
    parser.add_argument('--temperature', type=float, default=0.0, help="Sampling temperature (higher = more diverse, 0 = greedy)")
    parser.add_argument('--use-greedy', action='store_true', help="Use greedy decoding (ignores temperature)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_path):
        print(f"Error: Audio file not found at {args.audio_path}")
        return
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
    
    # Transcribe the audio
    result = transcribe(args.audio_path, args.model_path, args.temperature, args.use_greedy)
    
    # Print and save results
    print("\nTranscription result:")
    print(result)
    
    output_file = "transcription_result.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result)
    
    print(f"\nTranscription saved to {output_file}")

if __name__ == "__main__":
    main() 