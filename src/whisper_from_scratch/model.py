import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

from .blocks import ResidualAttentionBlock
from .feature_extraction import log_mel_spectrogram

class AudioEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, n_state, 3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, 3, stride=2, padding=1)
        
        self.positional_embedding = nn.Parameter(torch.randn(n_ctx, n_state))
        self.blocks = nn.ModuleList([
            ResidualAttentionBlock(n_state, n_head)
            for _ in range(n_layer)
        ])
        self.ln_post = nn.LayerNorm(n_state)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        
        x = x + self.positional_embedding[:x.shape[1], :]
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_post(x)
        return x

class TextDecoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.randn(n_ctx, n_state))
        
        self.blocks = nn.ModuleList([
            ResidualAttentionBlock(n_state, n_head, cross_attention=True)
            for _ in range(n_layer)
        ])
        
        self.ln = nn.LayerNorm(n_state)
        self.output = nn.Linear(n_state, n_vocab, bias=False)
        
    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.token_embedding(x)
        x = x + self.positional_embedding[:x.shape[1], :]
        
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            padding_mask = padding_mask.expand(-1, -1, x.shape[1], -1)
            
        for block in self.blocks:
            x = block(x, encoder_out, padding_mask)
            
        x = self.ln(x)
        x = self.output(x)
        return x

class Whisper(nn.Module):
    def __init__(
        self,
        dims: Dict,
    ):
        super().__init__()
        
        self.encoder = AudioEncoder(
            n_mels=dims["n_mels"],
            n_ctx=dims["n_audio_ctx"],
            n_state=dims["n_audio_state"],
            n_head=dims["n_audio_head"],
            n_layer=dims["n_audio_layer"]
        )
        
        self.decoder = TextDecoder(
            n_vocab=dims["n_vocab"],
            n_ctx=dims["n_text_ctx"],
            n_state=dims["n_text_state"],
            n_head=dims["n_text_head"],
            n_layer=dims["n_text_layer"]
        )
        
    def forward(
        self,
        audio: torch.Tensor,
        tokens: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        encoder_out = self.encoder(audio)
        decoder_out = self.decoder(tokens, encoder_out, padding_mask)
        return decoder_out
        
    @staticmethod
    def get_default_dims(size: str = "small") -> Dict:
        """Get model dimensions based on size."""
        sizes = {
            "tiny": {
                "n_mels": 80,
                "n_vocab": 51865,
                "n_audio_ctx": 1500,
                "n_audio_state": 384,
                "n_audio_head": 6,
                "n_audio_layer": 4,
                "n_text_ctx": 448,
                "n_text_state": 384,
                "n_text_head": 6,
                "n_text_layer": 4
            },
            "small": {
                "n_mels": 80,
                "n_vocab": 51865,
                "n_audio_ctx": 1500,
                "n_audio_state": 768,
                "n_audio_head": 12,
                "n_audio_layer": 12,
                "n_text_ctx": 448,
                "n_text_state": 768,
                "n_text_head": 12,
                "n_text_layer": 12
            }
        }
        return sizes[size]

    def save(self, path: str):
        """Save model state dict."""
        torch.save({
            "dims": self.get_default_dims(),
            "model_state_dict": self.state_dict()
        }, path)
    
    @classmethod
    def load(cls, path: str):
        """Load model from path."""
        checkpoint = torch.load(path)
        model = cls(checkpoint["dims"])
        model.load_state_dict(checkpoint["model_state_dict"])
        return model