"""Generate AAAI-style VEIL experiment figures from validated CSV tables."""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-veil-paper")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper" / "aaai2027" / "evidence"
FIGURES = ROOT / "paper" / "aaai2027" / "figures"
DATASETS = ("flowers", "caltech101", "dtd")
DATASET_LABELS = {
    "flowers": "Flowers102",
    "caltech101": "Caltech101",
    "dtd": "DTD",
}
METHODS = ("FedAvg", "Prompt-DP", "HAMP", "VEIL", "DP-FPL", "FedASK")
ATTACKS = (
    "fedmia_loss",
    "fedmia_cosine",
    "fedmia_joint",
    "nasr_passive",
    "rmia",
    "quantile_mia",
)
ATTACK_LABELS = ("Loss", "Cosine", "Joint", "Nasr", "RMIA", "Quantile")
COLORS = {
    "FedAvg": "#4B5563",
    "Prompt-DP": "#D97706",
    "HAMP": "#7C3AED",
    "VEIL": "#0369A1",
    "DP-FPL": "#9A3412",
    "FedASK": "#3F6212",
}
MARKERS = {
    "FedAvg": "o",
    "Prompt-DP": "s",
    "HAMP": "^",
    "VEIL": "D",
    "DP-FPL": "P",
    "FedASK": "X",
}


mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 28,
        "axes.titlesize": 32,
        "axes.labelsize": 29,
        "xtick.labelsize": 28,
        "ytick.labelsize": 28,
        "legend.fontsize": 28,
        "figure.titlesize": 34,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.edgecolor": "#374151",
        "axes.labelcolor": "#111827",
        "text.color": "#111827",
        "xtick.color": "#374151",
        "ytick.color": "#374151",
        "axes.grid": True,
        "grid.color": "#D1D5DB",
        "grid.linewidth": 1.2,
        "grid.alpha": 0.75,
    }
)
for _font_key in (
    "font.size",
    "axes.titlesize",
    "axes.labelsize",
    "xtick.labelsize",
    "ytick.labelsize",
    "legend.fontsize",
    "figure.titlesize",
):
    if float(mpl.rcParams[_font_key]) < 28:
        raise RuntimeError(f"Paper figure font {_font_key} is below 28 pt.")


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def privacy_utility(rows: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.8), sharey=True)
    shown = ("FedAvg", "Prompt-DP", "HAMP", "VEIL")
    global_ymax = 0.0
    for ax, dataset in zip(axes, DATASETS):
        x_low = float("inf")
        x_high = float("-inf")
        for method in shown:
            row = next(
                item
                for item in rows
                if item["dataset"] == dataset and item["method"] == method
            )
            ax.errorbar(
                float(row["accuracy_mean"]),
                float(row["worst_tpr_mean"]),
                xerr=float(row["accuracy_std"]),
                yerr=float(row["worst_tpr_std"]),
                fmt=MARKERS[method],
                markersize=16,
                markerfacecolor="white" if method != "VEIL" else COLORS[method],
                markeredgecolor=COLORS[method],
                markeredgewidth=3,
                ecolor=COLORS[method],
                elinewidth=2.4,
                capsize=7,
                label=method,
            )
            accuracy = float(row["accuracy_mean"])
            accuracy_std = float(row["accuracy_std"])
            x_low = min(x_low, accuracy - accuracy_std)
            x_high = max(x_high, accuracy + accuracy_std)
            global_ymax = max(
                global_ymax,
                float(row["worst_tpr_mean"]) + float(row["worst_tpr_std"]),
            )
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xlabel("Accuracy")
        margin = max(0.025, (x_high - x_low) * 0.12)
        ax.set_xlim(max(0.0, x_low - margin), min(1.0, x_high + margin))
    axes[0].set_ylim(0.0, global_ymax * 1.08)
    axes[0].set_ylabel("Worst-attack TPR@1%FPR")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.subplots_adjust(top=0.76, wspace=0.18)
    save(fig, "fedavg_privacy_utility.pdf")


def private_attack_profile(rows: list[dict[str, str]]) -> None:
    lookup = {
        (row["dataset"], row["method"], row["attack"]): float(row["tpr_mean"])
        for row in rows
    }
    x = np.arange(len(ATTACKS))
    width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(20, 7.2), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        for offset, method, hatch in (
            (-width / 2, "DP-FPL", "//"),
            (width / 2, "FedASK", "\\\\"),
        ):
            values = [lookup[(dataset, method, attack)] for attack in ATTACKS]
            ax.bar(
                x + offset,
                values,
                width,
                label=method,
                color=COLORS[method],
                edgecolor="#111827",
                linewidth=1.5,
                hatch=hatch,
            )
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xticks(x, ATTACK_LABELS, rotation=38, ha="right")
        ax.set_xlabel("Attack")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("TPR@1%FPR")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.subplots_adjust(top=0.78, bottom=0.25, wspace=0.16)
    save(fig, "private_methods_attack_profile.pdf")


def attack_heatmap(rows: list[dict[str, str]]) -> None:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["attack"])].append(float(row["tpr_mean"]))
    matrix = np.array(
        [
            [np.mean(grouped[(method, attack)]) for attack in ATTACKS]
            for method in METHODS
        ]
    )
    fig, ax = plt.subplots(figsize=(18, 9.5))
    image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=max(0.04, matrix.max()))
    ax.grid(False)
    ax.set_xticks(np.arange(len(ATTACKS)), ATTACK_LABELS, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(METHODS)), METHODS)
    ax.set_xlabel("Membership attack")
    ax.set_ylabel("Training / defense mechanism")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            red, green, blue, _alpha = image.cmap(image.norm(matrix[i, j]))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            color = "white" if luminance < 0.52 else "#111827"
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color=color)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    colorbar.set_label("Mean TPR@1%FPR")
    save(fig, "attack_defense_heatmap.pdf")


def ablation(rows: list[dict[str, str]]) -> None:
    variants = [row["variant"] for row in rows]
    accuracy = [float(row["accuracy"]) for row in rows]
    worst = [float(row["worst_tpr"]) for row in rows]
    mean_tpr = [float(row["mean_tpr"]) for row in rows]
    y = np.arange(len(variants))
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharey=True)
    accuracy_bars = axes[0].barh(
        y,
        accuracy,
        color="#0369A1",
        edgecolor="#111827",
        linewidth=1.5,
    )
    axes[0].set_xlabel("Accuracy")
    axes[0].set_xlim(0, max(accuracy) * 1.18)
    axes[0].bar_label(
        accuracy_bars,
        labels=[f"{value:.3f}" for value in accuracy],
        padding=6,
        fontsize=28,
    )
    axes[0].set_yticks(y, variants)
    axes[0].invert_yaxis()
    privacy_bars = axes[1].barh(
        y,
        worst,
        color="#D97706",
        edgecolor="#111827",
        linewidth=1.5,
        hatch="//",
    )
    axes[1].set_xlabel("Worst-attack TPR@1%FPR")
    axes[1].set_xlim(0, max(worst) * 1.30 if max(worst) > 0 else 1.0)
    axes[1].bar_label(
        privacy_bars,
        labels=[f"{value:.3f}" for value in worst],
        padding=6,
        fontsize=28,
    )
    axes[1].set_yticks(y, variants)
    axes[1].invert_yaxis()
    mean_bars = axes[2].barh(
        y,
        mean_tpr,
        color="#7C3AED",
        edgecolor="#111827",
        linewidth=1.5,
        hatch="xx",
    )
    axes[2].set_xlabel("Mean TPR@1%FPR")
    axes[2].set_xlim(0, max(mean_tpr) * 1.30 if max(mean_tpr) > 0 else 1.0)
    axes[2].bar_label(
        mean_bars,
        labels=[f"{value:.3f}" for value in mean_tpr],
        padding=6,
        fontsize=28,
    )
    axes[2].set_yticks(y, variants)
    axes[2].invert_yaxis()
    fig.subplots_adjust(left=0.42, hspace=0.48)
    save(fig, "veil_ablation.pdf")


def main() -> None:
    aggregate = read_csv("aggregate.csv")
    attacks = read_csv("attack_aggregate.csv")
    ablations = read_csv("ablation.csv")
    privacy_utility(aggregate)
    private_attack_profile(attacks)
    attack_heatmap(attacks)
    ablation(ablations)
    print(f"Wrote paper figures to {FIGURES}")


if __name__ == "__main__":
    main()
