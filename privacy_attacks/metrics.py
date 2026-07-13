import torch


def roc_curve(labels: torch.Tensor, scores: torch.Tensor):
    labels = labels.detach().flatten().to(dtype=torch.long, device="cpu")
    scores = scores.detach().flatten().to(dtype=torch.float64, device="cpu")
    if labels.numel() != scores.numel() or labels.numel() == 0:
        raise ValueError("labels and scores must be non-empty and have equal length.")
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("membership metrics require both members and non-members.")

    order = torch.argsort(scores, descending=True, stable=True)
    sorted_labels = labels[order]
    tps = torch.cumsum(sorted_labels == 1, dim=0).to(torch.float64)
    fps = torch.cumsum(sorted_labels == 0, dim=0).to(torch.float64)
    distinct = torch.ones_like(scores, dtype=torch.bool)
    distinct[:-1] = scores[order][:-1] != scores[order][1:]
    tpr = torch.cat((torch.zeros(1), tps[distinct] / positives, torch.ones(1)))
    fpr = torch.cat((torch.zeros(1), fps[distinct] / negatives, torch.ones(1)))
    return fpr, tpr


def membership_metrics(labels: torch.Tensor, scores: torch.Tensor) -> dict[str, float]:
    fpr, tpr = roc_curve(labels, scores)
    auc = float(torch.trapz(tpr, fpr).item())
    result = {"auc": auc}
    for target in (0.1, 0.01, 0.001):
        valid = tpr[fpr <= target]
        result[f"tpr_at_fpr_{target:g}"] = (
            float(valid.max().item()) if valid.numel() else 0.0
        )
    return result


def stratified_split(
    labels: torch.Tensor, calibration_fraction: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1.")
    generator = torch.Generator().manual_seed(seed)
    calibration = []
    evaluation = []
    for value in (0, 1):
        indices = torch.nonzero(labels == value, as_tuple=False).flatten()
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        count = min(max(1, int(indices.numel() * calibration_fraction)), indices.numel() - 1)
        calibration.append(indices[:count])
        evaluation.append(indices[count:])
    return torch.cat(calibration), torch.cat(evaluation)


def fit_linear_attack(
    features: torch.Tensor,
    labels: torch.Tensor,
    calibration_fraction: float,
    seed: int,
    epochs: int = 200,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train a deterministic supervised attack head and score held-out points."""
    features = features.detach().to(device="cpu", dtype=torch.float32)
    labels = labels.detach().to(device="cpu", dtype=torch.float32)
    calibration, evaluation = stratified_split(labels.long(), calibration_fraction, seed)
    train_x = features[calibration]
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    train_x = (train_x - mean) / std
    eval_x = (features[evaluation] - mean) / std

    torch.manual_seed(seed)
    model = torch.nn.Linear(features.shape[1], 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03, weight_decay=1e-3)
    target = labels[calibration].unsqueeze(1)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(train_x), target
        )
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        scores = torch.sigmoid(model(eval_x)).squeeze(1)
    return scores, labels[evaluation].long(), evaluation
