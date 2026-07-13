from collections.abc import Iterable

import torch
import torch.nn.functional as F


def trainable_names(model: torch.nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def clone_state(state: dict[str, torch.Tensor], cpu: bool = True):
    return {
        name: tensor.detach().cpu().clone() if cpu else tensor.detach().clone()
        for name, tensor in state.items()
    }


def flatten_state_delta(
    base_state: dict[str, torch.Tensor],
    updated_state: dict[str, torch.Tensor],
    names: Iterable[str],
) -> torch.Tensor:
    return torch.cat(
        [(base_state[name] - updated_state[name]).detach().flatten().cpu() for name in names]
    )


def _compress_vector(vector: torch.Tensor, max_size: int = 64) -> torch.Tensor:
    vector = vector.flatten()
    if vector.numel() <= max_size:
        return vector
    return F.adaptive_avg_pool1d(vector.view(1, 1, -1), max_size).flatten()


def gradient_signature(gradients: list[torch.Tensor]) -> torch.Tensor:
    """Prompt-aware white-box signature: token norms plus global statistics."""
    parts = []
    flat = []
    for gradient in gradients:
        detached = gradient.detach().cpu()
        flat.append(detached.flatten())
        if detached.ndim >= 2:
            token_norms = detached.reshape(-1, detached.shape[-1]).norm(dim=-1)
            parts.append(_compress_vector(token_norms))
        else:
            parts.append(_compress_vector(detached))
    all_values = torch.cat(flat)
    stats = torch.stack(
        (
            all_values.norm(),
            all_values.abs().mean(),
            all_values.std(unbiased=False),
            all_values.abs().max(),
        )
    )
    return torch.cat((*parts, stats))


def per_sample_prompt_gradients(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return flattened gradients, compact signatures, and losses per sample."""
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    flattened = []
    signatures = []
    losses = []
    model.eval()
    for image, label in zip(images, labels):
        model.zero_grad(set_to_none=True)
        logits = model(image.unsqueeze(0))
        loss = F.cross_entropy(logits, label.view(1))
        gradients = torch.autograd.grad(loss, parameters, retain_graph=False)
        flattened.append(torch.cat([gradient.detach().flatten().cpu() for gradient in gradients]))
        signatures.append(gradient_signature(list(gradients)))
        losses.append(loss.detach().cpu())
    return torch.stack(flattened), torch.stack(signatures), torch.stack(losses)


@torch.no_grad()
def logits_and_representation(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    try:
        output = model(images, return_intermediate=True)
    except TypeError:
        output = model(images)
    if isinstance(output, tuple) and len(output) == 3:
        logits, image_features, text_features = output
        conditioned = image_features * text_features[labels]
        representation = torch.cat((logits, conditioned), dim=1)
    else:
        logits = output[0] if isinstance(output, tuple) else output
        representation = logits
    losses = F.cross_entropy(logits, labels, reduction="none")
    compact = F.adaptive_avg_pool1d(
        representation.unsqueeze(1), min(64, representation.shape[1])
    ).squeeze(1)
    return logits.detach().cpu(), compact.detach().cpu(), losses.detach().cpu()
