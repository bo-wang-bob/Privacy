import torch
import torch.nn as nn
from typing import Tuple, List, Union, Optional, cast
from transformers import CLIPProcessor, CLIPModel
from transformers.modeling_attn_mask_utils import (
    _create_4d_causal_attention_mask,
    _prepare_4d_attention_mask,
)

from transformers.modeling_outputs import BaseModelOutput
from trainmodel.base_model import BaseModel

import logging

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import re
import torch.nn.functional as F


DEFAULT_PROMPT_TEMPLATE = "a photo of a {}"
CIFAR100_PROMPT_TEMPLATE = "a low-resolution photo of a {class}"
FLOWERS_PROMPT_TEMPLATE = (
    "a photo of the flower species {class}, "
    "with distinctive petals, color, and shape"
)
OXFORDPETS_PROMPT_TEMPLATE = "a photo of the pet breed {class}, a cat or dog"
DTD_PROMPT_TEMPLATE = "a photo of {class} texture"


def to_display_name(name: str) -> str:
    """Convert class name to display format by replacing underscores with spaces."""
    name = name.replace("_", " ")
    name = re.sub(r"\s+", " ", name).strip()
    return name


def get_default_prompt_template(dataset_name: Optional[str]) -> str:
    if not dataset_name:
        return DEFAULT_PROMPT_TEMPLATE

    normalized_dataset_name = dataset_name.lower()
    if normalized_dataset_name == "cifar100":
        return CIFAR100_PROMPT_TEMPLATE
    if normalized_dataset_name == "flowers":
        return FLOWERS_PROMPT_TEMPLATE
    if normalized_dataset_name == "oxfordpets":
        return OXFORDPETS_PROMPT_TEMPLATE
    if normalized_dataset_name in ("dtd", "describable-textures", "describabletextures"):
        return DTD_PROMPT_TEMPLATE
    return DEFAULT_PROMPT_TEMPLATE


def format_prompt_template(template: str, class_name: str) -> str:
    if "{class}" in template:
        return template.format(**{"class": class_name})
    return template.format(class_name)


def entropic_partial_transport(
    similarities: torch.Tensor,
    epsilon: float,
    transported_mass: float,
    max_iterations: int,
    threshold: float,
) -> torch.Tensor:
    """FedOTP's entropic unbalanced OT plan with one relaxed marginal.

    ``similarities`` has shape ``(B, M, N)`` for B class/image pairs,
    M visual patches and N prompts. The returned plan is detached in the model
    forward, matching the paper's alternating step that fixes the OT plan while
    optimizing prompt vectors.
    """
    if similarities.ndim != 3:
        raise ValueError("FedOTP similarities must have shape (batch, patches, prompts).")
    if epsilon <= 0 or max_iterations <= 0 or threshold <= 0:
        raise ValueError("FedOTP epsilon, max_iterations, and threshold must be positive.")
    if not 0 < transported_mass <= 1:
        raise ValueError("FedOTP transported_mass must be in (0, 1].")

    batch, patches, prompts = similarities.shape
    if patches <= 0 or prompts <= 0:
        raise ValueError("FedOTP requires at least one patch and one prompt.")
    source = similarities.new_full((batch, patches), 1.0 / patches)
    target = similarities.new_full(
        (batch, prompts), transported_mass / prompts
    )
    kernel = torch.exp(-(1.0 - similarities) / epsilon).clamp_min(
        torch.finfo(similarities.dtype).tiny
    )
    kernel_source = kernel / source.unsqueeze(-1)
    kernel_target = kernel.transpose(1, 2) / target.unsqueeze(-1)
    left = torch.ones_like(source)
    right = torch.ones_like(target)
    for _ in range(max_iterations):
        previous = right
        left = torch.minimum(
            1.0 / torch.bmm(kernel_source, right.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12),
            torch.ones_like(left),
        )
        right = 1.0 / torch.bmm(
            kernel_target, left.unsqueeze(-1)
        ).squeeze(-1).clamp_min(1e-12)
        if float((right - previous).abs().mean()) < threshold:
            break
    plan = left.unsqueeze(-1) * kernel * right.unsqueeze(-2)
    if not bool(torch.isfinite(plan).all()):
        raise FloatingPointError("FedOTP transport plan contains non-finite values.")
    return plan


class PromptLearner(nn.Module):
    """
    Learnable prompt module for CLIP text encoder.

    This module creates learnable context tokens that are inserted between
    the prefix (BOS token) and suffix (class name + EOS) of the text prompt.
    Context tokens are shared by all classes by default, or class-specific
    when class_specific_ctx=True.
    Only the context tokens (self.ctx) are trainable, while prefix and suffix
    embeddings are frozen.
    """

    # Declare buffer types to avoid type checker treating them as Optional[Tensor]
    token_prefix: torch.Tensor
    token_suffix: torch.Tensor
    input_ids: torch.Tensor
    suffix_attention_mask: torch.Tensor
    ctx_positions: torch.Tensor

    def __init__(
        self,
        clip_model: CLIPModel,
        processor: CLIPProcessor,
        classnames: List[str],
        n_ctx: int = 32,
        template: str = DEFAULT_PROMPT_TEMPLATE,
        class_specific_ctx: bool = False,
        parameterization: str = "full",
        low_rank: int = 4,
        low_rank_scaling: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        # self.clip_model = clip_model
        self.processor = processor
        self.classnames = classnames
        self.n_cls: int = len(classnames)
        self.n_ctx: int = n_ctx
        self.class_specific_ctx = class_specific_ctx
        self.parameterization = parameterization.lower()
        self.low_rank = int(low_rank)
        self.low_rank_scaling = float(low_rank_scaling)
        self.device = device

        # Get text embedding dimension from CLIP model config
        d: int = int(clip_model.text_model.config.hidden_size)

        # Initialize learnable context tokens with small random values
        if self.class_specific_ctx:
            ctx_shape = (self.n_cls, n_ctx, d)
        else:
            ctx_shape = (n_ctx, d)
        initial_ctx = torch.randn(*ctx_shape, device=device) * 0.02
        if self.parameterization in {"full", "promptfl"}:
            self.ctx = nn.Parameter(initial_ctx)
        elif self.parameterization == "fedotp":
            self.global_ctx = nn.Parameter(initial_ctx.clone())
            local_ctx = torch.empty_like(initial_ctx)
            nn.init.normal_(local_ctx, std=0.02)
            self.local_ctx = nn.Parameter(local_ctx)
        elif self.parameterization == "fedpgp":
            self.global_ctx = nn.Parameter(initial_ctx.clone())
            rank = min(self.low_rank, n_ctx, d)
            if rank <= 0:
                raise ValueError("FedPGP low_rank must be positive.")
            self.low_rank = rank
            if self.class_specific_ctx:
                u_shape = (self.n_cls, n_ctx, rank)
                v_shape = (self.n_cls, rank, d)
            else:
                u_shape = (n_ctx, rank)
                v_shape = (rank, d)
            self.fedpgp_u = nn.Parameter(torch.empty(*u_shape, device=device))
            self.fedpgp_v = nn.Parameter(torch.empty(*v_shape, device=device))
            nn.init.normal_(self.fedpgp_u, std=0.02)
            nn.init.normal_(self.fedpgp_v, std=0.02)
        elif self.parameterization == "dpfpl":
            self.global_ctx = nn.Parameter(initial_ctx.clone())
            self.local_ctx = nn.Parameter(torch.zeros_like(initial_ctx))
        elif self.parameterization == "fedask":
            rows = int(initial_ctx.numel() // initial_ctx.shape[-1])
            rank = min(self.low_rank, rows, int(initial_ctx.shape[-1]))
            if rank <= 0:
                raise ValueError("FedASK low_rank must be positive.")
            self.low_rank = rank
            self.register_buffer("base_ctx", initial_ctx.clone())
            self.fedask_A = nn.Parameter(
                torch.randn(rank, initial_ctx.shape[-1], device=device) * 0.02
            )
            self.fedask_B = nn.Parameter(torch.zeros(rows, rank, device=device))
            self.fedask_scaling = self.low_rank_scaling
        else:
            raise ValueError(
                "parameterization must be one of: full, promptfl, fedotp, "
                "fedpgp, dpfpl, fedask."
            )

        # logger.info(
        #     f"PromptLearner initialized: n_cls={self.n_cls}, n_ctx={n_ctx}, hidden_size={d}, device={device}"
        # )

        # Step 1: Construct full text for each class (used to get suffix token ids and EOS position)
        display_class_names = [to_display_name(name) for name in classnames]
        paper_prompt = self.parameterization in {"promptfl", "fedotp", "fedpgp"}
        if paper_prompt:
            placeholder = " ".join(["X"] * self.n_ctx)
            texts = [f"{placeholder} {name}." for name in display_class_names]
        else:
            texts = [format_prompt_template(template, name) for name in display_class_names]
        tok = processor(text=texts, padding=True, truncation=True, return_tensors="pt")  # type: ignore

        input_ids_local = cast(torch.Tensor, tok["input_ids"])  # (K, L)
        attention_mask_local = cast(torch.Tensor, tok["attention_mask"])  # (K, L)

        logger.debug(
            f"Tokenized input_ids shape: {input_ids_local.shape}, attention_mask shape: {attention_mask_local.shape}"
        )

        # Step 2: Convert fixed tokens to embeddings using CLIP's token embedding layer
        with torch.no_grad():
            token_emb: torch.Tensor = clip_model.text_model.embeddings.token_embedding(
                input_ids_local.to(device=device)
            )  # (K, L, d)

        # Step 3: Extract prefix - only take the first token (BOS/start token)
        token_prefix: torch.Tensor = token_emb[:, :1, :]  # (K, 1, d)

        # PromptFL/FedOTP/FedPGP replace placeholder tokens, as in CoOp. The
        # legacy full parameterization retains its historical behavior of
        # prepending learned tokens to the configured handcrafted template.
        suffix_start = 1 + self.n_ctx if paper_prompt else 1
        token_suffix: torch.Tensor = token_emb[:, suffix_start:, :]
        suffix_attention_mask = attention_mask_local[:, suffix_start:]

        logger.debug(
            f"token_prefix shape: {token_prefix.shape}, token_suffix shape: {token_suffix.shape}"
        )

        # Register fixed embeddings as buffers (non-trainable, moved to specified device)
        self.register_buffer("token_prefix", token_prefix.to(device))
        self.register_buffer("token_suffix", token_suffix.to(device))
        self.register_buffer("suffix_attention_mask", suffix_attention_mask.to(device))
        if paper_prompt:
            # Fixed placeholder-token prompt used as the task-agnostic anchor
            # by the FedPGP reference implementation.
            self.register_buffer(
                "paper_anchor_context",
                token_emb[:, 1 : 1 + self.n_ctx, :].to(device),
            )
        self.register_buffer(
            "ctx_positions",
            torch.arange(1, 1 + self.n_ctx, device=device, dtype=torch.long),
        )

    def get_learnable_token_positions(self) -> torch.Tensor:
        """Return absolute positions of learnable context tokens in the prompt."""
        return self.ctx_positions

    def effective_context(self) -> torch.Tensor:
        if self.parameterization in {"full", "promptfl"}:
            return self.ctx
        if self.parameterization == "fedotp":
            return (self.global_ctx + self.local_ctx) / 2.0
        if self.parameterization == "fedpgp":
            return self.global_ctx + torch.matmul(self.fedpgp_u, self.fedpgp_v)
        if self.parameterization == "dpfpl":
            return self.global_ctx + self.local_ctx
        adapter = self.fedask_scaling * (self.fedask_B @ self.fedask_A)
        return self.base_ctx + adapter.reshape_as(self.base_ctx)

    def forward(
        self, ctx_override: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct prompt embeddings by concatenating prefix, learnable context, and suffix.

        Returns:
            prompt_embeds: (K, 1+n_ctx+L-1, d) - Full prompt embeddings for all classes
            attn_mask: (K, 1+n_ctx+L-1) - Corresponding attention mask
        """
        K: int = self.n_cls

        effective_ctx = (
            self.effective_context() if ctx_override is None else ctx_override
        )
        if effective_ctx.ndim == 2:
            # Expand shared learnable context to all classes: (n_ctx, d) -> (K, n_ctx, d)
            ctx = effective_ctx.unsqueeze(0).expand(K, -1, -1)
        elif effective_ctx.ndim == 3 and effective_ctx.shape[0] == K:
            ctx = effective_ctx
        else:
            raise ValueError(
                "Prompt context must have shape (n_ctx, dim) or "
                "(num_classes, n_ctx, dim)."
            )

        # Concatenate: [BOS] + [learnable context] + [class name + EOS]
        prompt_embeds = torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)

        # Build attention mask: insert ones for the learnable context positions
        suffix_mask = self.suffix_attention_mask
        ctx_mask = torch.ones(
            K,
            self.n_ctx,
            device=self.device,
            dtype=suffix_mask.dtype,
        )  # (K, n_ctx) - all ones for learnable context
        prefix_mask = torch.ones(
            K,
            1,
            device=self.device,
            dtype=suffix_mask.dtype,
        )  # (K, 1) - one for BOS token

        # Concatenate masks in same order as embeddings
        attn_mask = torch.cat(
            [
                prefix_mask,
                ctx_mask,
                suffix_mask,
            ],
            dim=1,
        )  # (K, 1+n_ctx+L-1)

        logger.debug(
            f"PromptLearner forward: prompt_embeds shape={prompt_embeds.shape}, attn_mask shape={attn_mask.shape}"
        )
        return prompt_embeds, attn_mask


class TextEncoder(nn.Module):
    """
    Text encoder wrapper that processes prompt embeddings through CLIP's text transformer.

    Takes pre-constructed prompt embeddings (from PromptLearner) and produces
    text features aligned with image features in CLIP's joint embedding space.
    """

    def __init__(self, device: torch.device = torch.device("cpu")):
        super().__init__()
        # self.text_model = clip_model.text_model
        # self.text_projection = clip_model.text_projection
        self.device = device
        # logger.info(f"TextEncoder initialized: device={device}")

    def forward(
        self,
        prompt_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        clip_model: CLIPModel,
        return_last_attn: bool = False,
        return_all_attn: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode prompt embeddings to text features.

        Args:
            prompt_embeds: (K, Lp, d) - Prompt embeddings for K classes
            attention_mask: (K, Lp) - Attention mask for valid positions
            return_last_attn: If True, also return the last transformer layer
                attention weights.
            return_all_attn: If True, also return all transformer layer
                attention weights.

        Returns:
            text_embeds: (K, projection_dim) - Text features in CLIP's joint space
            last_attn: (K, num_heads, Lp, Lp) - Last transformer layer attention
                weights, returned only when return_last_attn=True
            all_attn: (num_layers, K, num_heads, Lp, Lp) - All transformer
                layer attention weights, returned only when return_all_attn=True
        """
        if return_last_attn and return_all_attn:
            raise ValueError(
                "return_last_attn and return_all_attn cannot both be True"
            )

        K, Lp, _ = prompt_embeds.shape
        position_ids = torch.arange(Lp, device=prompt_embeds.device).unsqueeze(0)

        # Step 1: Add positional embeddings to prompt embeddings
        pos_emb = clip_model.text_model.embeddings.position_embedding(
            position_ids
        )  # (1, Lp, d)
        inputs_embeds = prompt_embeds + pos_emb  # (K, Lp, d)

        # Create causal attention mask for autoregressive text modeling
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape=(K, Lp),
            dtype=prompt_embeds.dtype,
            device=prompt_embeds.device,
        )

        # Prepare 4D attention mask if not using flash attention
        attn_mask_4d = None
        if attention_mask is not None and not getattr(
            clip_model.text_model, "_use_flash_attention_2", False
        ):
            attn_mask_4d = _prepare_4d_attention_mask(
                attention_mask, prompt_embeds.dtype
            )

        # Step 2: Pass through transformer encoder
        out: BaseModelOutput = clip_model.text_model.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask_4d,
            causal_attention_mask=causal_attention_mask,
            output_attentions=(return_last_attn or return_all_attn),
            output_hidden_states=False,
        )

        last_hidden = out.last_hidden_state  # (K, Lp, d)
        assert last_hidden is not None

        K = last_hidden.size(0)

        # Step 3: Apply final layer normalization
        last_hidden = clip_model.text_model.final_layer_norm(last_hidden)

        # Step 4: Extract EOS token hidden state as pooled output
        # EOS position is the last valid token (sum of attention mask - 1)
        eos_pos = attention_mask.sum(dim=-1) - 1  # (K,)
        pooled = last_hidden[torch.arange(K, device=self.device), eos_pos]  # (K, d)

        # Step 5: Project to CLIP's joint embedding space
        text_embeds = clip_model.text_projection(pooled)  # (K, projection_dim)

        if return_all_attn:
            attentions = out.attentions
            if attentions is None or len(attentions) == 0:
                raise RuntimeError(
                    "Attention weights are unavailable for the current text encoder configuration. "
                    "Load CLIPModel with attn_implementation='eager' to enable output_attentions."
                )
            all_attn = torch.stack([cast(torch.Tensor, attn) for attn in attentions], dim=0)
            logger.debug(
                f"TextEncoder forward: input shape=({K}, {Lp}), text_embeds shape={text_embeds.shape}, "
                f"all_attn shape={all_attn.shape}"
            )
            return text_embeds, all_attn

        if return_last_attn:
            attentions = out.attentions
            if attentions is None or len(attentions) == 0 or attentions[-1] is None:
                raise RuntimeError(
                    "Attention weights are unavailable for the current text encoder configuration. "
                    "Load CLIPModel with attn_implementation='eager' to enable output_attentions."
                )
            last_attn = cast(torch.Tensor, attentions[-1])
            logger.debug(
                f"TextEncoder forward: input shape=({K}, {Lp}), text_embeds shape={text_embeds.shape}, "
                f"last_attn shape={last_attn.shape}"
            )
            return text_embeds, last_attn

        logger.debug(
            f"TextEncoder forward: input shape=({K}, {Lp}), text_embeds shape={text_embeds.shape}"
        )
        return text_embeds


class CustomCLIP(BaseModel):
    """
    Custom CLIP model with learnable text prompts for few-shot/zero-shot classification.

    This model freezes the original CLIP weights and only trains the learnable
    context tokens in the prompt, enabling efficient adaptation to new tasks.

    Architecture:
        - Frozen CLIP image encoder: Extracts image features
        - PromptLearner: Generates learnable prompt embeddings
        - TextEncoder: Encodes prompts to text features
        - Classification: Cosine similarity between image and text features
    """

    _supports_native_hamp_output = True

    def __init__(
        self,
        clip_model: CLIPModel,
        processor: CLIPProcessor,
        classnames: list,
        n_ctx=32,
        template: str = DEFAULT_PROMPT_TEMPLATE,
        class_specific_ctx: bool = False,
        parameterization: str = "full",
        low_rank: int = 4,
        low_rank_scaling: float = 1.0,
        method_config: Optional[dict] = None,
        device=torch.device("cpu"),
    ):
        super().__init__()

        # Move CLIP model to device and freeze all parameters
        self.clip_model = clip_model.to(device)  # type: ignore
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.clip_model.eval()

        self.classnames = classnames
        self.n_ctx: int = n_ctx
        self.template = template
        self.class_specific_ctx = class_specific_ctx
        self.parameterization = parameterization.lower()
        self.low_rank = int(low_rank)
        self.low_rank_scaling = float(low_rank_scaling)
        self.method_config = dict(method_config or {})
        self.processor = processor
        self.device = device

        # Initialize learnable prompt module
        self.prompt_learner = PromptLearner(
            clip_model=self.clip_model,
            processor=self.processor,
            classnames=self.classnames,
            n_ctx=self.n_ctx,
            template=self.template,
            class_specific_ctx=self.class_specific_ctx,
            parameterization=self.parameterization,
            low_rank=self.low_rank,
            low_rank_scaling=self.low_rank_scaling,
            device=self.device,
        )

        # Initialize text encoder wrapper
        self.text_encoder = TextEncoder(device=self.device)

        self.fedotp_epsilon = float(self.method_config.get("epsilon", 0.01))
        self.fedotp_transported_mass = float(
            self.method_config.get("transported_mass", 0.8)
        )
        self.fedotp_max_iterations = int(
            self.method_config.get("max_iterations", 100)
        )
        self.fedotp_threshold = float(self.method_config.get("threshold", 1e-3))
        self.fedpgp_contrastive_weight = float(
            self.method_config.get("contrastive_weight", 0.5)
        )
        self.fedpgp_temperature = float(
            self.method_config.get("temperature", 0.5)
        )
        if self.parameterization == "fedpgp":
            with torch.no_grad():
                anchor_features = self._text_features_for_context(
                    self.prompt_learner.paper_anchor_context
                )
            self.register_buffer("fedpgp_anchor_features", anchor_features)

        # logger.info(
        #     f"CustomCLIP initialized: n_classes={len(classnames)}, n_ctx={n_ctx}, device={device}"
        # )
        logger.debug(f"CustomCLIP classnames: {classnames}")

    def forward_with_context(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        return_intermediate: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass for image classification.

        Args:
            x: (B, C, H, W) - Batch of input images

        Returns:
            logits: (B, K) - Classification logits for K classes
            image_features: (B, projection_dim) - Image features
            text_features: (K, projection_dim) - Text features
        """
        x = x.to(self.device)

        if self.parameterization == "fedotp" and context is None:
            return self._forward_fedotp(x, return_intermediate=return_intermediate)

        # Extract image features using frozen CLIP image encoder
        image_features = self.clip_model.get_image_features(pixel_values=x)  # type: ignore
        # L2 normalize image features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Generate text features from learnable prompts
        prompt_embeds, attn_mask = self.prompt_learner(context)
        text_features = self.text_encoder(
            prompt_embeds, attn_mask, self.clip_model
        )
        # (K, projection_dim)
        # L2 normalize text features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity scaled by learned temperature
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()  # (B, K)
        if not self.training:
            logits = logits / float(getattr(self, "_hamp_output_temperature", 1.0))

        logger.debug(
            f"CustomCLIP forward: batch_size={x.shape[0]}, logits shape={logits.shape}, logit_scale={logit_scale.item():.4f}"
        )
        if return_intermediate:
            return logits, image_features, text_features
        return logits

    def _text_features_for_context(self, context: torch.Tensor) -> torch.Tensor:
        prompt_embeds, attn_mask = self.prompt_learner(context)
        return F.normalize(
            self.text_encoder(prompt_embeds, attn_mask, self.clip_model), dim=-1
        )

    def _forward_fedotp(
        self,
        x: torch.Tensor,
        return_intermediate: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """FedOTP global/local prompt cooperation over CLIP patch features."""
        vision = self.clip_model.vision_model(pixel_values=x)
        last_hidden = cast(torch.Tensor, vision.last_hidden_state)
        pooled = cast(torch.Tensor, vision.pooler_output)
        image_features = F.normalize(self.clip_model.visual_projection(pooled), dim=-1)
        patch_hidden = last_hidden[:, 1:, :]
        patch_hidden = self.clip_model.vision_model.post_layernorm(patch_hidden)
        patch_features = F.normalize(
            self.clip_model.visual_projection(patch_hidden), dim=-1
        )

        learner = self.prompt_learner
        global_text = self._text_features_for_context(learner.global_ctx)
        local_text = self._text_features_for_context(learner.local_ctx)
        text_features = torch.stack((global_text, local_text), dim=0)

        # (B, C, M, N), then solve one UOT problem per image/class pair.
        similarities = torch.einsum(
            "bmd,ncd->bcmn", patch_features, text_features
        )
        batch, classes, patches, prompts = similarities.shape
        flat = similarities.reshape(batch * classes, patches, prompts)
        with torch.no_grad():
            plan = entropic_partial_transport(
                flat,
                epsilon=self.fedotp_epsilon,
                transported_mass=self.fedotp_transported_mass,
                max_iterations=self.fedotp_max_iterations,
                threshold=self.fedotp_threshold,
            )
        scores = (plan * flat).sum(dim=(1, 2)).reshape(batch, classes)
        logits = self.clip_model.logit_scale.exp() * scores
        if not self.training:
            logits = logits / float(getattr(self, "_hamp_output_temperature", 1.0))
        if return_intermediate:
            return logits, image_features, F.normalize(
                text_features.mean(dim=0), dim=-1
            )
        return logits

    def fedpgp_contrastive_loss(self) -> torch.Tensor:
        """Prompt-wise contrastive objective from the FedPGP paper."""
        if self.parameterization != "fedpgp":
            raise ValueError("fedpgp_contrastive_loss requires FedPGP parameterization.")
        learner = self.prompt_learner
        global_text = self._text_features_for_context(learner.global_ctx)
        personalized_text = self._text_features_for_context(
            learner.effective_context()
        )
        positive = F.cosine_similarity(
            self.fedpgp_anchor_features, global_text, dim=-1
        )
        negative = F.cosine_similarity(global_text, personalized_text, dim=-1)
        logits = torch.stack((positive, negative), dim=-1) / self.fedpgp_temperature
        targets = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, targets)

    def fedpgp_training_loss(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        logits = cast(torch.Tensor, self(images))
        return F.cross_entropy(logits, labels) + (
            self.fedpgp_contrastive_weight * self.fedpgp_contrastive_loss()
        )

    def forward(self, x: torch.Tensor, return_intermediate: bool = False):
        return self.forward_with_context(
            x, context=None, return_intermediate=return_intermediate
        )

    def get_effective_prompt(self) -> torch.Tensor:
        """Return the prompt matrix actually consumed by the text encoder."""
        return self.prompt_learner.effective_context()

    @torch.no_grad()
    def get_text_features(self, normalize: bool = True) -> torch.Tensor:
        """
        Return the text encoder output for every class.

        Args:
            normalize: If True, L2-normalize the projected text features.

        Returns:
            text_features: (K, projection_dim) - one vector per class.
        """
        if self.parameterization == "fedotp":
            learner = self.prompt_learner
            text_features = (
                self._text_features_for_context(learner.global_ctx)
                + self._text_features_for_context(learner.local_ctx)
            ) / 2.0
        else:
            prompt_embeds, attn_mask = self.prompt_learner()
            text_features = self.text_encoder(
                prompt_embeds, attn_mask, self.clip_model
            )
        if normalize:
            text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return text_features

    @torch.no_grad()
    def get_token_attention(self) -> torch.Tensor:
        """
        Return raw EOS-query attention over learnable context tokens.

        For every class prompt, this extracts the attention from the EOS query
        token to each learnable context-token key for every transformer layer
        and attention head.

        Returns:
            ctx_to_eos_attention: (num_layers, K, num_heads, n_ctx)
        """
        prompt_embeds, attn_mask = self.prompt_learner()
        _, all_attn = cast(
            Tuple[torch.Tensor, torch.Tensor],
            self.text_encoder(
                prompt_embeds,
                attn_mask,
                self.clip_model,
                return_all_attn=True,
            ),
        )

        # all_attn: (N, K, H, Lp, Lp)
        num_layers, num_classes, num_heads, _, prompt_len = all_attn.shape
        eos_positions = (attn_mask.sum(dim=-1) - 1).long()  # (K,)
        ctx_positions = (
            self.prompt_learner.get_learnable_token_positions().long()
        )  # (n_ctx,)
        # Select the EOS query row for each class while preserving layer/head axes.
        eos_index = eos_positions.view(1, num_classes, 1, 1, 1).expand(
            num_layers,
            num_classes,
            num_heads,
            1,
            prompt_len,
        )
        eos_query_attn = all_attn.gather(
            dim=3,
            index=eos_index,
        ).squeeze(3)  # (N, K, H, Lp)
        ctx_to_eos_attention = eos_query_attn.index_select(
            dim=-1,
            index=ctx_positions,
        )  # (N, K, H, n_ctx)

        logger.debug(
            "CustomCLIP attention analysis: "
            f"all_attn={all_attn.shape}, prompt_len={prompt_len}, "
            f"ctx_to_eos_attention={ctx_to_eos_attention.shape}"
        )
        return ctx_to_eos_attention

    @torch.no_grad()
    def clone_prompt_only(self):
        new_model = CustomCLIP(
            clip_model=self.clip_model,
            processor=self.processor,
            classnames=self.classnames,
            n_ctx=self.n_ctx,
            template=self.template,
            class_specific_ctx=self.class_specific_ctx,
            parameterization=self.parameterization,
            low_rank=self.low_rank,
            low_rank_scaling=self.low_rank_scaling,
            method_config=self.method_config,
            device=self.device,
        )
        new_model.prompt_learner.load_state_dict(
            self.prompt_learner.state_dict(), strict=True
        )
        if hasattr(self, "_hamp_output_temperature"):
            new_model._hamp_output_temperature = self._hamp_output_temperature
        return new_model

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        new_obj = self.clone_prompt_only()
        memo[id(self)] = new_obj
        return new_obj
