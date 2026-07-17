"""Generate the paper's focused AAAI-style attack comparison figure."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-veil-paper")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper" / "aaai2027" / "evidence"
FIGURES = ROOT / "paper" / "aaai2027" / "figures"

DATASETS = ("flowers", "caltech101", "dtd")
DATASET_LABELS = {
    "flowers": "Flowers102",
    "caltech101": "Caltech101",
    "dtd": "DTD",
}
METHODS = ("VEIL", "DP-FPL", "FedASK")
ATTACKS = (
    "fedmia_loss",
    "fedmia_cosine",
    "fedmia_joint",
    "nasr_passive",
    "rmia",
    "quantile_mia",
)
ATTACK_LABELS = (
    "Loss",
    "Cos.",
    "Joint",
    "Nasr",
    "RMIA",
    "Quant.",
)
TEXT_COLOR = "#252525"
VEIL_COLOR = "#1F5F8B"
PALETTE = ("#F7FAFC", "#BFD8E8", "#2F6FA3")


def setup_style() -> None:
    """Match the serif typography and restrained palette of the reference plot."""

    from matplotlib import font_manager

    for style in ("Regular", "Bold", "Italic", "BoldItalic"):
        path = Path(f"/usr/share/fonts/truetype/croscore/Tinos-{style}.ttf")
        if path.exists():
            font_manager.fontManager.addfont(str(path))
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Tinos", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 25,
            "axes.titlesize": 30,
            "axes.labelsize": 27,
            "xtick.labelsize": 25,
            "ytick.labelsize": 27,
            "figure.titlesize": 32,
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    checked = (
        "font.size",
        "axes.titlesize",
        "axes.labelsize",
        "xtick.labelsize",
        "ytick.labelsize",
        "figure.titlesize",
    )
    for key in checked:
        if float(mpl.rcParams[key]) < 25:
            raise RuntimeError(f"Paper figure font {key} is below 25 pt.")


def read_attack_rows() -> list[dict[str, str]]:
    path = DATA / "attack_aggregate.csv"
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def comparison_heatmaps(rows: list[dict[str, str]]) -> None:
    """Plot VEIL, DP-FPL, and FedASK under every audited attack."""

    lookup = {
        (row["dataset"], row["method"], row["attack"]): float(row["tpr_mean"])
        for row in rows
    }
    matrices = {
        dataset: np.asarray(
            [
                [lookup[(dataset, method, attack)] for attack in ATTACKS]
                for method in METHODS
            ]
        )
        for dataset in DATASETS
    }
    vmax = max(float(matrix.max()) for matrix in matrices.values())
    cmap = LinearSegmentedColormap.from_list("veil_blue", PALETTE, N=256)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17.0, 5.2),
        sharey=True,
        facecolor="white",
        constrained_layout=True,
    )
    image = None
    for panel_index, (axis, dataset) in enumerate(zip(axes, DATASETS)):
        matrix = matrices[dataset]
        percentage_matrix = 100.0 * matrix
        image = axis.imshow(
            percentage_matrix,
            cmap=cmap,
            vmin=0.0,
            vmax=100.0 * vmax,
            interpolation="nearest",
            aspect="auto",
        )
        axis.set_title(DATASET_LABELS[dataset], pad=12, fontweight="normal")
        axis.set_xticks(np.arange(len(ATTACKS)), ATTACK_LABELS)
        axis.set_yticks(np.arange(len(METHODS)), METHODS)
        axis.tick_params(axis="x", length=0, rotation=22)
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")
        axis.tick_params(
            axis="y",
            length=0,
            labelleft=panel_index == 0,
            pad=10,
        )
        axis.set_xticks(np.arange(-0.5, len(ATTACKS), 1.0), minor=True)
        axis.set_yticks(np.arange(-0.5, len(METHODS), 1.0), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.4)
        axis.tick_params(which="minor", bottom=False, left=False)
        if panel_index == 0:
            axis.set_ylabel("Training mechanism")

        # A blue outline identifies our method without distorting the color scale.
        axis.add_patch(
            Rectangle(
                (-0.5, -0.5),
                len(ATTACKS),
                1.0,
                fill=False,
                edgecolor=VEIL_COLOR,
                linewidth=3.0,
                clip_on=False,
            )
        )
        if panel_index == 0:
            axis.get_yticklabels()[0].set_color(VEIL_COLOR)
            axis.get_yticklabels()[0].set_fontweight("bold")

        for spine in axis.spines.values():
            spine.set_linewidth(0.8)
            spine.set_edgecolor("#4A4A4A")

    if image is None:
        raise RuntimeError("No attack-comparison panels were generated.")
    colorbar = fig.colorbar(
        image,
        ax=axes,
        location="right",
        fraction=0.028,
        pad=0.02,
    )
    colorbar.set_label("TPR@1% FPR (%)")
    colorbar.ax.tick_params(labelsize=25, width=1.0)
    colorbar.outline.set_linewidth(0.7)

    FIGURES.mkdir(parents=True, exist_ok=True)
    output = FIGURES / "private_methods_attack_profile.pdf"
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {output}")


def main() -> None:
    setup_style()
    comparison_heatmaps(read_attack_rows())


if __name__ == "__main__":
    main()
