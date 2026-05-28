"""Anonymized per-evaluator agreeableness figures for the paper.

Re-creates the same content as
`distribution_tests/figures/_evaluator_agreeableness_*.png` but maps real
designer names to Designer A through Designer E inside each cohort.
The mapping is by mean pairwise tau ASCENDING within each cohort, so
Designer A is the most-disagreeing (outlier) evaluator and Designer E is
the most-agreeing.  Figures are written to the --out-dir as
`eval_agree_aes.png` and `eval_agree_desc.png`.

Run via:
    python make_anon_eval_figures.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

HERE = Path(__file__).parent
DT = HERE.parent  # distribution_tests/
sys.path.insert(0, str(DT))

from taste_stats import DIMENSIONS, load_rank_tensor, compute_stats, DATA_DIR
from evaluator_agreeableness import per_evaluator_tau_arrays

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--out-dir", default=str(DT / "figures"),
                 help="Directory to write the generated figures into.")
_args, _ = _ap.parse_known_args()
OUT = Path(_args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)

# Short axis labels for the heatmap (matches Fig 1/2 abbreviations).
# DIMENSIONS' display strings are long ("UI+Ad Preference (holistic)") and
# crash into each other when rotated at 25 degrees in the half-column
# display size the paper uses.
SHORT_DIM_NAMES = {
    "aesthetics_preference":    "UI+Ad Pref",
    "aesthetics_mood":          "Mood & Tone",
    "aesthetics_visual_hier":   "Visual Hier.",
    "aesthetics_color_harmony": "Color Harm.",
    "aesthetics_typography":    "Typography",
    "descriptions_preference":  "Pref.",
    "descriptions_color_acc":   "Color Acc.",
    "descriptions_spatial_acc": "Spatial Acc.",
    "descriptions_typography":  "Typography",
}


def collect_cohort_matrix(group_name, group_dims):
    """Return (eval_names_alpha, dim_displays, mat_shape_evaluators_x_dims)."""
    matrix = {}
    dim_displays = []
    eval_names = None
    for slug, display, group, csv_name in group_dims:
        tensor, _, evaluators = load_rank_tensor(
            str(DATA_DIR / csv_name)
        )
        stats = compute_stats(tensor, evaluators)
        per_eval_arr = per_evaluator_tau_arrays(stats)
        per_eval_mean = {e: float(a.mean()) for e, a in per_eval_arr.items()}
        if eval_names is None:
            eval_names = sorted(per_eval_mean.keys())
        dim_displays.append(SHORT_DIM_NAMES.get(slug, display))
        for e in eval_names:
            matrix.setdefault(e, []).append(per_eval_mean[e])
    mat = np.array([matrix[e] for e in eval_names])
    return eval_names, dim_displays, mat


LABEL_PREFIX = {"Aesthetics": "A", "Descriptions": "D"}


def render_anon(group_name, eval_names, dim_displays, mat, out_path):
    """Render the 2-panel anonymized figure (heatmap + bar chart).

    Anonymisation rule: sort evaluators by overall mean tau ascending and
    relabel within the cohort.  Aesthetics designers are A1 to A5;
    Descriptions designers are D1 to D5.  Index 1 is the lowest-agreement
    rater in the cohort; index 5 is the highest.  The two cohorts use
    disjoint designer pools, so A1 and D1 are different people.
    """
    overall = mat.mean(axis=1)
    order_asc = np.argsort(overall)  # outlier first
    prefix = LABEL_PREFIX[group_name]
    anon_labels = [f"{prefix}{i + 1}" for i in range(len(eval_names))]
    sorted_overall = overall[order_asc]
    sorted_mat = mat[order_asc]

    n_eval, n_dim = sorted_mat.shape
    # Smaller figsize since each cohort figure is displayed side-by-side
    # in LaTeX at ~0.46\linewidth; we need to survive ~50% scaling.
    fig, axes = plt.subplots(
        1, 2, figsize=(7.5, 2.5 + 0.35 * n_eval),
        gridspec_kw={"width_ratios": [1.7, 1.0], "wspace": 0.55},
    )

    sns.heatmap(
        sorted_mat, ax=axes[0], annot=True, fmt="+.2f", cmap="RdYlGn",
        vmin=-0.10, vmax=0.40,
        cbar_kws={"label": r"mean pairwise $\tau$"},
        xticklabels=dim_displays, yticklabels=anon_labels,
        annot_kws={"fontsize": 9},
    )
    axes[0].set_title(
        f"{group_name}: per-evaluator $\\tau$ vs. peers",
        fontsize=12,
    )
    axes[0].tick_params(axis="x", rotation=25, labelsize=11)
    axes[0].tick_params(axis="y", rotation=0, labelsize=11)
    # bump colorbar label and tick size
    cbar = axes[0].collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(r"mean pairwise $\tau$", fontsize=11)

    ax = axes[1]
    bar_y = np.arange(n_eval)
    cmap = plt.cm.RdYlGn
    bar_colors = cmap((sorted_overall + 0.10) / 0.50)
    bars = ax.barh(bar_y, sorted_overall, color=bar_colors, edgecolor="black")
    ax.set_yticks(bar_y)
    ax.set_yticklabels(anon_labels, fontsize=11)
    ax.invert_yaxis()
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel(r"overall mean $\tau$", fontsize=12)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_title(
        f"Overall ({group_name})",
        fontsize=12,
    )
    for j, b in enumerate(bars):
        v = sorted_overall[j]
        ax.text(
            v + 0.005, b.get_y() + b.get_height() / 2,
            f"{v:+.3f}", va="center", fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")
    print(f"  anonymisation key: ", end="")
    for orig_idx, anon_label in zip(order_asc, anon_labels):
        print(f"({anon_label} = real index {int(orig_idx)})", end=" ")
    print()


def main():
    aes_dims = [(s, d, g, c) for s, d, g, c in DIMENSIONS if g == "Aesthetics"]
    desc_dims = [(s, d, g, c) for s, d, g, c in DIMENSIONS if g == "Descriptions"]

    print("Aesthetics cohort:")
    eval_names, dim_displays, mat = collect_cohort_matrix("Aesthetics", aes_dims)
    render_anon(
        "Aesthetics", eval_names, dim_displays, mat,
        OUT / "eval_agree_aes.png",
    )

    print("\nDescriptions cohort:")
    eval_names, dim_displays, mat = collect_cohort_matrix(
        "Descriptions", desc_dims
    )
    render_anon(
        "Descriptions", eval_names, dim_displays, mat,
        OUT / "eval_agree_desc.png",
    )


if __name__ == "__main__":
    main()
