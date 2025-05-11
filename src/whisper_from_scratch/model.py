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
        # Extract current dimensions from model
        n_audio_state = self.encoder.positional_embedding.shape[1]
        n_text_state = self.decoder.positional_embedding.shape[1]
        
        # Determine model size
        model_size = "custom"
        if n_audio_state == 384:
            model_size = "tiny"
        elif n_audio_state == 768:
            model_size = "small"
        elif n_audio_state == 1280:
            model_size = "medium"
        
        # Get full dimensions
        dims = {
            "n_mels": 80,
            "n_vocab": self.decoder.token_embedding.weight.shape[0],
            "n_audio_ctx": self.encoder.positional_embedding.shape[0],
            "n_audio_state": n_audio_state,
            "n_audio_head": self.encoder.blocks[0].attn.n_head,
            "n_audio_layer": len(self.encoder.blocks),
            "n_text_ctx": self.decoder.positional_embedding.shape[0],
            "n_text_state": n_text_state,
            "n_text_head": self.decoder.blocks[0].attn.n_head,
            "n_text_layer": len(self.decoder.blocks)
        }
        
        # Save checkpoint with dimensions
        torch.save({
            "model_state_dict": self.state_dict(),
            "dims": dims,
            "model_size": model_size
        }, path)
    
    @classmethod
    def load(cls, path: str):
        """Load model from path."""
        checkpoint = torch.load(path)
        
        # Get model dimensions from checkpoint
        model_dims = checkpoint.get("dims")
        
        if model_dims is None:
            # Try to infer dimensions from model state dict
            state_dict = checkpoint["model_state_dict"]
            embed_shape = None
            
            # Check encoder's positional embedding to determine model size
            for key, param in state_dict.items():
                if 'encoder.positional_embedding' in key:
                    embed_shape = param.shape[1]
                    break
            
            if embed_shape == 384:
                model_dims = cls.get_default_dims("tiny")
            elif embed_shape == 768:
                model_dims = cls.get_default_dims("small")
            elif embed_shape == 1280:
                model_dims = cls.get_default_dims("medium")
            else:
                raise ValueError(f"Could not determine model dimensions from checkpoint")
        
        print(f"Loading model with dimensions: {model_dims}")
        model = cls(model_dims)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model