import logging
import os

import torch

from context.context import Context
from utils.seismograph_analysis_common import (
    log_average_pairwise_metric,
    save_matrix_csv,
)

logger = logging.getLogger(__name__)


def build_user_update_vectors(
    ctx: Context,
    user_ids: list[int],
    device: torch.device,
) -> torch.Tensor:
    update_vectors = []
    for user_id in user_ids:
        base_model_dict = ctx.get_base_model_state(user_id)
        updated_model_dict = ctx.get_updated_model_state(user_id)
        update_parts = []
        for name in ctx.trainable_param_names:
            update_parts.append(
                (updated_model_dict[name] - base_model_dict[name])
                .detach()
                .to(device=device, dtype=torch.float32)
                .flatten()
            )
        update_vectors.append(torch.cat(update_parts))

    return torch.stack(update_vectors, dim=0)


def build_user_update_matrices(
    ctx: Context,
    user_ids: list[int],
    device: torch.device,
) -> list[torch.Tensor]:
    update_matrices = []
    for user_id in user_ids:
        base_model_dict = ctx.get_base_model_state(user_id)
        updated_model_dict = ctx.get_updated_model_state(user_id)
        user_update_matrices = []
        for name in ctx.trainable_param_names:
            update = (updated_model_dict[name] - base_model_dict[name]).detach().to(
                device=device,
                dtype=torch.float32,
            )
            if update.dim() == 2:
                user_update_matrices.append(update)
            elif update.dim() > 2:
                user_update_matrices.append(update.reshape(update.shape[0], -1))
            else:
                user_update_matrices.append(update.reshape(1, -1))

        if len(user_update_matrices) == 1:
            update_matrices.append(user_update_matrices[0])
        else:
            # FPL normally has one matrix parameter: prompt_learner.ctx,
            # shaped (n_ctx, embedding_dim). This fallback keeps other
            # trainable layouts analyzable without assuming shared widths.
            flattened_update = torch.cat(
                [matrix.flatten() for matrix in user_update_matrices]
            )
            update_matrices.append(flattened_update.reshape(1, -1))

    return update_matrices


def compute_pairwise_update_metrics(
    update_vectors: torch.Tensor,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        l1_distance = torch.cdist(update_vectors, update_vectors, p=1)
        l2_distance = torch.cdist(update_vectors, update_vectors, p=2)

        normalized_updates = torch.nn.functional.normalize(
            update_vectors,
            p=2,
            dim=1,
            eps=1e-12,
        )
        cosine_similarity = torch.mm(
            normalized_updates,
            normalized_updates.transpose(0, 1),
        ).clamp(-1.0, 1.0)
        cosine_distance = 1.0 - cosine_similarity

        signs = torch.sign(update_vectors)
        num_users = signs.shape[0]
        sign_consistency = torch.empty(
            (num_users, num_users),
            device=update_vectors.device,
            dtype=torch.float32,
        )
        for row_idx in range(num_users):
            sign_consistency[row_idx] = (
                signs[row_idx].unsqueeze(0) == signs
            ).float().mean(dim=1)

        diagonal_indices = torch.arange(
            update_vectors.shape[0],
            device=update_vectors.device,
        )
        l1_distance[diagonal_indices, diagonal_indices] = 0.0
        l2_distance[diagonal_indices, diagonal_indices] = 0.0
        cosine_distance[diagonal_indices, diagonal_indices] = 0.0
        sign_consistency[diagonal_indices, diagonal_indices] = 1.0

    return {
        "l1_distance": l1_distance,
        "l2_distance": l2_distance,
        "cosine_distance": cosine_distance,
        "sign_consistency": sign_consistency,
    }


def compute_update_matrix_top1_singular_value_distance(
    update_matrices: list[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    singular_values = []
    with torch.no_grad():
        for matrix in update_matrices:
            singular_values.append(torch.linalg.svdvals(matrix)[0])

        values = torch.stack(singular_values).to(
            device=device,
            dtype=torch.float32,
        )
        distance_matrix = torch.abs(values.unsqueeze(0) - values.unsqueeze(1))
        diagonal_indices = torch.arange(
            distance_matrix.shape[0],
            device=distance_matrix.device,
        )
        distance_matrix[diagonal_indices, diagonal_indices] = 0.0
        return distance_matrix


def save_pairwise_update_metric_matrices(
    ctx: Context,
    user_ids: list[int],
    device: torch.device,
) -> None:
    if not user_ids:
        logger.warning(
            "SeismographAggregator: no users available for update metric analysis."
        )
        return
    if not ctx.trainable_param_names:
        logger.warning(
            "SeismographAggregator: no trainable parameters available for update metric analysis."
        )
        return

    update_vectors = build_user_update_vectors(ctx, user_ids, device)
    update_matrices = build_user_update_matrices(ctx, user_ids, device)
    metric_matrices = compute_pairwise_update_metrics(update_vectors)
    metric_matrices["matrix_top1_singular_value_distance"] = (
        compute_update_matrix_top1_singular_value_distance(update_matrices, device)
    )
    metric_dirs = {
        "l1_distance": "update_l1_distance",
        "l2_distance": "update_l2_distance",
        "cosine_distance": "update_cosine_distance",
        "sign_consistency": "update_sign_consistency",
        "matrix_top1_singular_value_distance": "update_matrix_top1_singular_value_distance",
    }

    for metric_name, matrix in metric_matrices.items():
        log_average_pairwise_metric(
            round_idx=ctx.glob_iter,
            metric_name=metric_name,
            matrix=matrix,
            user_ids=user_ids,
        )
        csv_path = os.path.join(
            ctx.results_dir,
            metric_dirs[metric_name],
            f"round_{ctx.glob_iter}.csv",
        )
        save_matrix_csv(matrix, user_ids, csv_path)

    logger.info(
        "SeismographAggregator: saved pairwise update metric matrices for round %s.",
        ctx.glob_iter,
    )
