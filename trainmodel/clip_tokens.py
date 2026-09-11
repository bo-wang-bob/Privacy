"""Explicit CLIP input-token forward, preserving all downstream PEFT gradients."""
import torch
from torch.nn import functional as F


def token_features(clip_model, tokens):
    vision = clip_model.vision_model
    expected = vision.embeddings.num_positions
    width = clip_model.config.vision_config.hidden_size
    if tokens.ndim != 3 or tokens.shape[1:] != (expected, width):
        raise ValueError(f"Expected CLIP input tokens [B, {expected}, {width}].")
    hidden = vision.pre_layrnorm(tokens)
    encoded = vision.encoder(inputs_embeds=hidden)
    pooled = vision.post_layernorm(encoded.last_hidden_state[:, 0])
    return clip_model.visual_projection(pooled).float()


class CLIPTokenInputMixin:
    @torch.no_grad()
    def encode_input_tokens(self, images):
        return self.clip_model.vision_model.embeddings(images.to(self.device)).detach()

    def forward_tokens(self, tokens):
        image = F.normalize(token_features(self.clip_model, tokens.to(self.device)), dim=-1)
        text = F.normalize(self.encode_class_texts(), dim=-1)
        return self.clip_model.logit_scale.exp().detach() * image @ text.T


class ClientCLIPTokenInputMixin:
    def encode_input_tokens(self, images):
        return self._call_shared("encode_input_tokens", images)

    def forward_tokens(self, tokens):
        return self._call_shared("forward_tokens", tokens)
