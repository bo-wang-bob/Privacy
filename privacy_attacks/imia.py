import copy

import torch
import torch.nn.functional as F

from privacy_attacks.base import AttackResult
from privacy_attacks.model_utils import (
    balanced_evaluation_indices,
    imitative_weights,
    probabilities_for,
    reset_trainable_parameters,
    scaled_confidence,
    train_cross_entropy,
)


def _select_pivots(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    auxiliary: torch.Tensor,
    per_class: int,
) -> torch.Tensor:
    selected = []
    for value in labels[auxiliary].unique().tolist():
        candidates = auxiliary[labels[auxiliary] == value]
        losses = -probabilities[candidates, int(value)].clamp_min(1e-12).log()
        order = candidates[torch.argsort(losses)]
        selected.append(order[: max(1, min(per_class, order.numel()))])
    return torch.cat(selected) if selected else auxiliary


def _train_imitation(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    target_probabilities: torch.Tensor,
    steps: int,
    learning_rate: float,
) -> None:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=learning_rate)
    weights = imitative_weights(target_probabilities, labels)
    model.train()
    for _ in range(max(1, steps)):
        optimizer.zero_grad()
        probabilities = torch.softmax(model(images), dim=1).clamp_min(1e-12)
        loss = (
            weights * (probabilities.log() - target_probabilities.log()).square()
        ).sum(dim=1).mean()
        loss.backward()
        optimizer.step()


def run_imia(
    target_model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    membership: torch.Tensor,
    auxiliary_fraction: float,
    seed: int,
    imitative_models: int = 4,
    warmup_steps: int = 10,
    imitation_steps: int = 20,
    pivot_steps: int = 20,
    learning_rate: float = 0.02,
    pivots_per_class: int = 4,
) -> AttackResult:
    """Non-adaptive IMIA using target-informed parameter-efficient models."""
    auxiliary, evaluation = balanced_evaluation_indices(
        membership, auxiliary_fraction, seed
    )
    device = images.device
    auxiliary_device = auxiliary.to(device)
    evaluation_device = evaluation.to(device)
    labels_device = labels.to(device)
    with torch.no_grad():
        target_probabilities = probabilities_for(target_model, images).detach()
    pivots = _select_pivots(
        target_probabilities.detach().cpu(),
        labels.detach().cpu(),
        auxiliary,
        pivots_per_class,
    )
    pivots_device = pivots.to(device)
    out_scores = []
    in_scores_by_class: dict[int, list[torch.Tensor]] = {}
    generator = torch.Generator().manual_seed(seed)

    for model_index in range(max(1, imitative_models)):
        bootstrap_positions = torch.randint(
            auxiliary.numel(),
            (auxiliary.numel(),),
            generator=generator,
        )
        bootstrap = auxiliary[bootstrap_positions].to(device)
        model = copy.deepcopy(target_model).to(device)
        reset_trainable_parameters(model, seed + 100 + model_index)
        train_cross_entropy(
            model,
            images[bootstrap],
            labels_device[bootstrap],
            warmup_steps,
            learning_rate,
        )
        warm_model = copy.deepcopy(model)

        _train_imitation(
            model,
            images[auxiliary_device],
            labels_device[auxiliary_device],
            target_probabilities[auxiliary_device],
            imitation_steps,
            learning_rate,
        )
        out_probabilities = probabilities_for(model, images[evaluation_device])
        out_scores.append(
            scaled_confidence(out_probabilities, labels_device[evaluation_device])
            .detach()
            .cpu()
        )

        in_model = warm_model.to(device)
        train_cross_entropy(
            in_model,
            images[pivots_device],
            labels_device[pivots_device],
            pivot_steps,
            learning_rate,
        )
        pivot_probabilities = probabilities_for(in_model, images[pivots_device])
        pivot_scores = scaled_confidence(
            pivot_probabilities, labels_device[pivots_device]
        ).detach().cpu()
        pivot_labels = labels[pivots].detach().cpu()
        for value in pivot_labels.unique().tolist():
            in_scores_by_class.setdefault(int(value), []).append(
                pivot_scores[pivot_labels == value]
            )

    out_mean = torch.stack(out_scores).mean(dim=0)
    global_in = torch.cat(
        [part for parts in in_scores_by_class.values() for part in parts]
    ).mean()
    in_mean = []
    for value in labels[evaluation].tolist():
        parts = in_scores_by_class.get(int(value))
        in_mean.append(torch.cat(parts).mean() if parts else global_in)
    in_mean = torch.stack(in_mean)
    observed = scaled_confidence(
        target_probabilities[evaluation_device], labels_device[evaluation_device]
    ).detach().cpu()
    scores = (observed - out_mean).square() - (observed - in_mean).square()
    return AttackResult(
        name="imia",
        scores=scores,
        labels=membership[evaluation].detach().cpu(),
        sample_indices=evaluation.detach().cpu(),
        metadata={
            "imitative_models": max(1, imitative_models),
            "warmup_steps": warmup_steps,
            "imitation_steps": imitation_steps,
            "pivot_steps": pivot_steps,
            "pivot_samples": int(pivots.numel()),
            "prompt_only_models": (
                str(getattr(target_model, "model_type", "")) != "clip_mlp"
            ),
            "trainable_scope": (
                "mlp_only"
                if str(getattr(target_model, "model_type", "")) == "clip_mlp"
                else "prompt_only"
            ),
        },
    )
