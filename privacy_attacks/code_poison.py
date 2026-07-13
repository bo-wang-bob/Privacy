import hashlib

import torch
import torch.nn.functional as F

from privacy_attacks.base import AttackResult


def _seed_from_sample(sample: torch.Tensor) -> int:
    raw = sample.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().tobytes()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63 - 1)


def generate_membership_encoding_samples(
    images: torch.Tensor,
    mean: float = 0.0,
    std: float = 0.1,
) -> torch.Tensor:
    """Deterministically map each candidate to its secret synthetic partner."""
    encoded = []
    for image in images:
        generator = torch.Generator(device="cpu").manual_seed(_seed_from_sample(image))
        noise = torch.randn(image.shape, generator=generator, dtype=torch.float32)
        encoded.append((noise * std + mean).to(dtype=image.dtype))
    return torch.stack(encoded).to(images.device)


def compromised_prompt_loss(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    weight: float,
    mean: float,
    std: float,
) -> torch.Tensor:
    clean_loss = F.cross_entropy(model(images), labels)
    synthetic = generate_membership_encoding_samples(images, mean=mean, std=std)
    secret_loss = F.cross_entropy(model(synthetic), labels)
    return clean_loss + weight * secret_loss


@torch.no_grad()
def run_code_poison_attack(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    membership: torch.Tensor,
    mean: float,
    std: float,
) -> AttackResult:
    model.eval()
    synthetic = generate_membership_encoding_samples(images, mean=mean, std=std)
    losses = F.cross_entropy(model(synthetic), labels, reduction="none")
    indices = torch.arange(labels.numel())
    return AttackResult(
        name="codepoison",
        scores=(-losses).detach().cpu(),
        labels=membership.detach().cpu(),
        sample_indices=indices,
        metadata={"synthetic_mean": mean, "synthetic_std": std},
    )
