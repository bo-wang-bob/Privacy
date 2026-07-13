import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """Minimal interface shared by prompt-tuned models."""

    def forward(self, x, return_intermediate: bool = False):
        raise NotImplementedError

    def get_text_features(self, normalize: bool = True) -> torch.Tensor:
        raise NotImplementedError

    def get_token_attention(self) -> torch.Tensor:
        raise NotImplementedError
