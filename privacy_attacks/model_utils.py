import copy
import math

import torch
import torch.nn.functional as F


def last_client_states(
    observations: list[dict], target_client_id: int
) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]], int]:
    """Return the most recent target state and same-round reference states."""
    for observation in reversed(observations):
        states = observation.get("client_states", {})
        if target_client_id in states:
            references = [
                state for client_id, state in states.items()
                if client_id != target_client_id
            ]
            return states[target_client_id], references, int(observation["round"])
    raise ValueError("No stored prompt state is available for the target client.")


def model_from_state(
    base_model: torch.nn.Module,
    state: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.nn.Module:
    model = copy.deepcopy(base_model).to(device)
    model.load_state_dict(state, strict=False)
    return model


def reset_trainable_parameters(
    model: torch.nn.Module, seed: int, std: float = 0.02
) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                values = torch.randn(
                    parameter.shape,
                    generator=generator,
                    dtype=parameter.dtype,
                    device="cpu",
                ).to(parameter.device)
                parameter.copy_(values * std)


def train_cross_entropy(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    steps: int,
    learning_rate: float,
) -> None:
    if steps <= 0 or labels.numel() == 0:
        return
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=learning_rate)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(images), labels)
        loss.backward()
        optimizer.step()


def true_class_probability(
    probabilities: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    return probabilities.gather(1, labels.view(-1, 1)).squeeze(1)


def scaled_confidence(
    probabilities: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """IMIA/LiRA log true-class probability minus best wrong-class log probability."""
    probabilities = probabilities.clamp_min(1e-12)
    true = true_class_probability(probabilities, labels)
    masked = probabilities.clone()
    masked.scatter_(1, labels.view(-1, 1), -1.0)
    wrong = masked.max(dim=1).values.clamp_min(1e-12)
    return true.log() - wrong.log()


@torch.no_grad()
def probabilities_for(
    model: torch.nn.Module, images: torch.Tensor
) -> torch.Tensor:
    model.eval()
    return torch.softmax(model(images), dim=1)


@torch.no_grad()
def semantic_features(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract image and label-conditioned prompt features without using predictions."""
    if hasattr(model, "clip_model") and hasattr(model, "get_text_features"):
        image_features = model.clip_model.get_image_features(pixel_values=images)
        text_features = model.get_text_features(normalize=True)
        image_features = F.normalize(image_features, dim=1)
        return image_features, text_features[labels]

    output = model(images, return_intermediate=True)
    if not isinstance(output, tuple) or len(output) != 3:
        raise TypeError(
            "PIPRA/PromptMIA need a model that returns image and prompt features."
        )
    _, image_features, text_features = output
    return F.normalize(image_features, dim=1), F.normalize(text_features[labels], dim=1)


def balanced_evaluation_indices(
    membership: torch.Tensor,
    auxiliary_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split known OUT samples into auxiliary and held-out sets; keep all IN queries."""
    if not 0.0 < auxiliary_fraction < 1.0:
        raise ValueError("auxiliary_fraction must be between zero and one.")
    out_indices = torch.nonzero(membership == 0, as_tuple=False).flatten()
    in_indices = torch.nonzero(membership == 1, as_tuple=False).flatten()
    if out_indices.numel() < 2 or in_indices.numel() == 0:
        raise ValueError("The attack needs members and at least two known non-members.")
    generator = torch.Generator().manual_seed(seed)
    order = out_indices[torch.randperm(out_indices.numel(), generator=generator)]
    count = min(max(1, int(order.numel() * auxiliary_fraction)), order.numel() - 1)
    auxiliary = order[:count]
    evaluation = torch.cat((in_indices, order[count:]))
    return auxiliary, evaluation


def imitative_weights(
    target_probabilities: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Class weights from IMIA Equation (2)."""
    classes = target_probabilities.shape[1]
    root = math.sqrt(classes)
    low = 1.0 / (classes + 2.0 * root)
    high = (1.0 + root) / (classes + 2.0 * root)
    weights = torch.full_like(target_probabilities, low)
    weights.scatter_(1, labels.view(-1, 1), high)
    masked = target_probabilities.clone()
    masked.scatter_(1, labels.view(-1, 1), -1.0)
    wrong = masked.argmax(dim=1, keepdim=True)
    weights.scatter_(1, wrong, high)
    return weights
