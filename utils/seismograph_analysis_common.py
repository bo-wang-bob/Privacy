import csv
import logging
import os

import torch

logger = logging.getLogger(__name__)


def save_matrix_csv(
    matrix: torch.Tensor,
    user_ids: list[int],
    csv_path: str,
) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    matrix_values = matrix.detach().cpu().tolist()
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["user_id", *user_ids])
        for user_id, row in zip(user_ids, matrix_values):
            writer.writerow([user_id, *[f"{value:.10g}" for value in row]])


def average_pairwise_metric_values(matrix: torch.Tensor) -> torch.Tensor:
    num_users = matrix.shape[0]
    if num_users <= 1:
        return torch.zeros(
            num_users,
            device=matrix.device,
            dtype=torch.float32,
        )
    matrix_values = matrix.detach().to(dtype=torch.float32)
    return (matrix_values.sum(dim=1) - matrix_values.diagonal()) / (num_users - 1)


def log_average_pairwise_metric(
    *,
    round_idx: int,
    metric_name: str,
    matrix: torch.Tensor,
    user_ids: list[int],
) -> torch.Tensor:
    if matrix.shape[0] != len(user_ids):
        logger.warning(
            "SeismographAggregator: %s matrix/user count mismatch at round %s.",
            metric_name,
            round_idx,
        )
        return torch.empty(0, device=matrix.device, dtype=torch.float32)

    average_values = average_pairwise_metric_values(matrix)
    logger.info(
        "SeismographAggregator: round %s %s average pairwise values: %s",
        round_idx,
        metric_name,
        [
            (user_id, round(float(value), 10))
            for user_id, value in zip(
                user_ids,
                average_values.detach().cpu().tolist(),
            )
        ],
    )
    return average_values
