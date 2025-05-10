import torch
import torch.nn.functional as F
import numpy as np
import librosa

def pad_or_trim(signal, length: int = 480000, *, axis: int = -1):
    """Pad or trim the audio signal to a fixed length."""
    if signal.shape[axis] > length:
        signal = signal.take(indices=range(length), axis=axis)

    if signal.shape[axis] < length:
        pad_widths = [(0, 0)] * signal.ndim
        pad_widths[axis] = (0, length - signal.shape[axis])
        signal = np.pad(signal, pad_widths)

    return signal

def log_mel_spectrogram(
    audio: torch.Tensor,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
    sample_rate: int = 16000,
    mel_min: float = 0.0,
    mel_max: float = 8000.0,
):
    """Convert audio waveform to log-mel spectrogram."""
    # Make sure audio is 2D (batch_size, signal) and float32
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    
    # Ensure float32 type
    audio = audio.to(torch.float32)
    
    # Compute mel spectrogram
    window = torch.hann_window(n_fft).to(audio.device)
    stft = torch.stft(audio, n_fft, hop_length, window=window, return_complex=True)
    magnitudes = stft.abs() ** 2

    # Mel filter bank
    mel_fb = torch.from_numpy(
        librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=mel_min,
            fmax=mel_max,
        )
    ).to(audio.device).to(torch.float32)  # Ensure float32

    # Convert to mel scale
    mel_spec = torch.matmul(mel_fb, magnitudes)
    
    # Convert to log scale
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    
    return log_spec