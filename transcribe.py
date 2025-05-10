import argparse
import os
import torch
import torchaudio
import numpy as np
from jiwer import wer
from tqdm import tqdm
import gc

from whisper_from_scratch.model import Whisper
from whisper_from_scratch.tokenizer import WhisperTokenizer
from whisper_from_scratch.feature_extraction import log_mel_spectrogram, pad_or_trim

def transcribe_audio(audio_path, model, tokenizer, device='cuda'):
    """Transcribe a single audio file."""
    # Load and preprocess audio
    print(f"Processing: {audio_path}")
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        
        # Convert to mono and normalize
        waveform = waveform.mean(dim=0)
        waveform = pad_or_trim(waveform.numpy())
        
        # Extract features
        mel = log_mel_spectrogram(torch.from_numpy(waveform))
        
        # Ensure features have correct shape [1, 80, seq_len]
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        if mel.shape[1] != 80:
            mel = mel.transpose(1, 2)
        if mel.shape[2] > 1500:
            mel = mel[:, :, :1500]
        
        # Move to device
        mel = mel.to(device)
        
        # Initialize decoder input with start token
        decoder_input = torch.tensor([[tokenizer.bos_token_id]], device=device)
        
        # Generate transcript
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                encoder_out = model.encoder(mel)
                
                # Generate up to 448 tokens (max context length)
                for _ in range(448):
                    # Forward pass through decoder
                    output = model.decoder(decoder_input, encoder_out)
                    
                    # Get next token
                    next_token = output[:, -1:].argmax(dim=-1)
                    
                    # If EOS token is generated, stop
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                        
                    # Append token and continue
                    decoder_input = torch.cat([decoder_input, next_token], dim=1)
        
        # Decode tokens to text
        transcript = tokenizer.decode(decoder_input[0].cpu().tolist())
        
        # Clear memory
        del mel, encoder_out, output, decoder_input
        torch.cuda.empty_cache()
        gc.collect()
        
        return transcript
    
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return ""

def batch_transcribe(audio_paths, model, tokenizer, batch_size=4, device='cuda'):
    """Transcribe a batch of audio files."""
    results = []
    
    for i in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[i:i+batch_size]
        batch_results = []
        
        for path in batch_paths:
            transcript = transcribe_audio(path, model, tokenizer, device)
            batch_results.append(transcript)
            
        results.extend(batch_results)
    
    return results

def evaluate_dataset(audio_dir, reference_file, model, tokenizer, device='cuda'):
    """Evaluate model on a dataset with reference transcripts."""
    # Load reference texts
    references = {}
    with open(reference_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                file_id = parts[0]
                text = parts[1]
                references[file_id] = text
    
    # Get audio files
    audio_paths = []
    reference_texts = []
    
    for file_id, text in references.items():
        audio_path = os.path.join(audio_dir, f"{file_id}.wav")
        if os.path.exists(audio_path):
            audio_paths.append(audio_path)
            reference_texts.append(text)
    
    # Transcribe all files
    predictions = batch_transcribe(audio_paths, model, tokenizer, device=device)
    
    # Calculate WER
    error_rate = wer(reference_texts, predictions)
    print(f"Word Error Rate: {error_rate:.4f}")
    
    # Return detailed results
    results = []
    for i, (ref, pred) in enumerate(zip(reference_texts, predictions)):
        results.append({
            "file": os.path.basename(audio_paths[i]),
            "reference": ref,
            "prediction": pred,
            "wer": wer([ref], [pred])
        })
    
    return error_rate, results

def transcribe_directory(audio_dir, output_file, model, tokenizer, device='cuda'):
    """Transcribe all audio files in a directory."""
    # Get all audio files
    audio_paths = []
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac')):
                audio_paths.append(os.path.join(root, file))
    
    print(f"Found {len(audio_paths)} audio files")
    
    # Transcribe all files
    results = []
    for path in tqdm(audio_paths):
        transcript = transcribe_audio(path, model, tokenizer, device)
        results.append({
            "file": os.path.relpath(path, audio_dir),
            "transcript": transcript
        })
    
    # Write results to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(f"{item['file']}\t{item['transcript']}\n")
    
    print(f"Transcriptions saved to {output_file}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Whisper transcription script")
    parser.add_argument("--model_path", default="whisper-mongolian-from-scratch/best_model.pt", help="Path to the trained model")
    parser.add_argument("--mode", choices=["single", "batch", "evaluate", "directory"], default="single", help="Transcription mode")
    parser.add_argument("--audio_path", help="Path to audio file for single transcription")
    parser.add_argument("--audio_dir", help="Directory containing audio files")
    parser.add_argument("--output_file", default="transcriptions.txt", help="Output file for batch transcription")
    parser.add_argument("--reference_file", help="Tab-separated file with reference transcriptions")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of CUDA")
    args = parser.parse_args()

    # Set device
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"Using device: {device}")

    # Load model and tokenizer
    print(f"Loading model from {args.model_path}")
    model = Whisper.load(args.model_path)
    model = model.to(device)
    model.eval()
    
    tokenizer = WhisperTokenizer()
    
    # Execute according to mode
    if args.mode == "single":
        if not args.audio_path:
            parser.error("--audio_path is required for single mode")
        transcript = transcribe_audio(args.audio_path, model, tokenizer, device)
        print(f"Transcription: {transcript}")
    
    elif args.mode == "batch":
        if not args.audio_dir:
            parser.error("--audio_dir is required for batch mode")
        
        audio_paths = []
        for root, _, files in os.walk(args.audio_dir):
            for file in files:
                if file.endswith(('.wav', '.mp3', '.flac')):
                    audio_paths.append(os.path.join(root, file))
        
        results = batch_transcribe(audio_paths, model, tokenizer, device=device)
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for i, result in enumerate(results):
                f.write(f"{os.path.basename(audio_paths[i])}\t{result}\n")
                
        print(f"Transcriptions saved to {args.output_file}")
    
    elif args.mode == "evaluate":
        if not args.audio_dir or not args.reference_file:
            parser.error("--audio_dir and --reference_file are required for evaluate mode")
            
        error_rate, results = evaluate_dataset(args.audio_dir, args.reference_file, model, tokenizer, device)
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for item in results:
                f.write(f"{item['file']}\t{item['reference']}\t{item['prediction']}\t{item['wer']:.4f}\n")
                
        print(f"Evaluation results saved to {args.output_file}")
    
    elif args.mode == "directory":
        if not args.audio_dir:
            parser.error("--audio_dir is required for directory mode")
            
        transcribe_directory(args.audio_dir, args.output_file, model, tokenizer, device)

if __name__ == "__main__":
    main() 