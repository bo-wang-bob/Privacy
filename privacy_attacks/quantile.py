import torch

from privacy_attacks.base import AttackResult
from privacy_attacks.model_utils import balanced_evaluation_indices


class QuantileRegressor(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        hidden_dim = max(4, min(hidden_dim, 128))
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(1)


def _pinball_loss(
    prediction: torch.Tensor, target: torch.Tensor, quantile: float
) -> torch.Tensor:
    residual = target - prediction
    return torch.maximum(quantile * residual, (quantile - 1.0) * residual).mean()


def run_quantile_mia(
    observations: list[dict],
    membership: torch.Tensor,
    labels: torch.Tensor,
    target_client_id: int,
    auxiliary_fraction: float,
    seed: int,
    quantile: float = 0.9,
    epochs: int = 200,
    learning_rate: float = 0.01,
) -> AttackResult:
    """NeurIPS QMIA adapted to prompt-model confidence and CLIP representations."""
    if not 0.0 < quantile < 1.0:
        raise ValueError("QMIA quantile must be between zero and one.")
    auxiliary, evaluation = balanced_evaluation_indices(
        membership, auxiliary_fraction, seed
    )
    for observation in reversed(observations):
        client_ids = observation["client_ids"].tolist()
        if target_client_id not in client_ids:
            continue
        position = client_ids.index(target_client_id)
        probabilities = observation["probabilities"][position].to(torch.float32)
        true_confidence = probabilities.gather(
            1, labels.detach().cpu().long().view(-1, 1)
        ).squeeze(1)
        features = observation["representations"][position].to(torch.float32)
        mean = features[auxiliary].mean(dim=0, keepdim=True)
        std = features[auxiliary].std(
            dim=0, keepdim=True, unbiased=False
        ).clamp_min(1e-6)
        features = (features - mean) / std

        torch.manual_seed(seed)
        regressor = QuantileRegressor(
            features.shape[1], min(64, max(8, features.shape[1]))
        )
        optimizer = torch.optim.Adam(
            regressor.parameters(), lr=learning_rate, weight_decay=1e-4
        )
        for _ in range(max(1, epochs)):
            optimizer.zero_grad()
            predicted = regressor(features[auxiliary])
            loss = _pinball_loss(
                predicted, true_confidence[auxiliary], quantile
            )
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            threshold = regressor(features[evaluation])
            scores = true_confidence[evaluation] - threshold
        return AttackResult(
            name="quantile_mia",
            scores=scores,
            labels=membership[evaluation].detach().cpu(),
            sample_indices=evaluation.detach().cpu(),
            metadata={
                "round": int(observation["round"]),
                "quantile": quantile,
                "auxiliary_nonmembers": int(auxiliary.numel()),
                "regression_epochs": max(1, epochs),
            },
        )
    raise ValueError("Quantile-MIA did not observe the target client.")
