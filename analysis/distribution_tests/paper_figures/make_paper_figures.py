"""Generate the 3 new paper figures for the TASTE paper.

Outputs (written to the --out-dir, default distribution_tests/figures/):
  fig_cross_domain.png    - per-dim T-distribution overlaid on Sushi/MovieLens/HPSv2 anchors
  fig_cycle_rates.png     - per-dim cycle rates with null line + anchor lines
  fig_subdim_signal.png   - per-dim mean pair-tau sorted descending, color-coded by cohort

Reads:
  ../stats/<slug>/stats.npz  - per-dim per-prompt stats
  ../refs/<ds>_*.npy          - reference-anchor arrays
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
DT = HERE.parent  # distribution_tests/
STATS = DT / "stats"
REFS = DT / "refs"
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--out-dir", default=str(DT / "figures"),
                 help="Directory to write the generated figures into.")
_args, _ = _ap.parse_known_args()
OUT = Path(_args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)

# Order: Aesthetics first (5 dims), Descriptions next (4 dims).
DIMS = [
    ("aesthetics_preference",    "UI+Ad Pref",      "Aesthetics"),
    ("aesthetics_mood",          "Mood & Tone",     "Aesthetics"),
    ("aesthetics_visual_hier",   "Visual Hier.",    "Aesthetics"),
    ("aesthetics_color_harmony", "Color Harmony",   "Aesthetics"),
    ("aesthetics_typography",    "Typography (A)",  "Aesthetics"),
    ("descriptions_preference",  "Desc Pref",       "Descriptions"),
    ("descriptions_color_acc",   "Color Acc.",      "Descriptions"),
    ("descriptions_spatial_acc", "Spatial Acc.",    "Descriptions"),
    ("descriptions_typography",  "Typography (D)",  "Descriptions"),
]

COHORT_COLOR = {
    "Aesthetics":   "#1f77b4",
    "Descriptions": "#2ca02c",
}
REF_COLOR = {
    "Sushi":      "#1f77b4",
    "MovieLens":  "#2ca02c",
    "HPSv2-test": "#ff7f0e",
}


def load_dim_stats():
    out = {}
    for slug, display, group in DIMS:
        d = np.load(STATS / slug / "stats.npz")
        out[slug] = {
            "T":        d["T"],
            "pair_tau": d["pair_tau"],
            "pmaj":     d["pmaj"],
            "cycle":    d["cycle"],
            "display":  display,
            "group":    group,
        }
    return out


def load_refs():
    out = {}
    for name in ["sushi", "movielens", "hpsv2"]:
        out[name] = {
            "T":        np.load(REFS / f"{name}_T.npy"),
            "pair_tau": np.load(REFS / f"{name}_pairtau.npy"),
            "pmaj":     np.load(REFS / f"{name}_pmaj.npy"),
            "cycle":    np.load(REFS / f"{name}_cycle.npy"),
        }
    return out


# ---- Figure 1: cross-domain T positioning ----------------------------------

def fig_cross_domain(stats, refs):
    """Violin/strip per TASTE dim of per-prompt T, with horizontal anchor
    lines for reference dataset T-medians."""
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    positions = np.arange(len(DIMS))
    parts = ax.violinplot(
        [stats[slug]["T"] for slug, _, _ in DIMS],
        positions=positions, widths=0.8, showmeans=False,
        showmedians=True, showextrema=False,
    )
    for pc, (slug, _, group) in zip(parts["bodies"], DIMS):
        pc.set_facecolor(COHORT_COLOR[group])
        pc.set_alpha(0.65)
        pc.set_edgecolor("black")
        pc.set_linewidth(0.6)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.0)

    # reference medians as horizontal lines spanning the violin region
    refs_T_median = {
        "Sushi":      float(np.median(refs["sushi"]["T"])),
        "MovieLens":  float(np.median(refs["movielens"]["T"])),
        "HPSv2-test": float(np.median(refs["hpsv2"]["T"])),
    }
    for name, v in refs_T_median.items():
        ax.axhline(v, color=REF_COLOR[name], linestyle="--", linewidth=1.0,
                   alpha=0.85, label=f"{name} median ({v:+.2f})")
    # null mean (0)
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.7, alpha=0.7,
               label="null mean (0)")

    # cohort separator
    ax.axvline(4.5, color="black", linewidth=0.4, alpha=0.4)

    ax.set_xticks(positions)
    ax.set_xticklabels([d[1] for d in DIMS], rotation=30, ha="right",
                       fontsize=11)
    # labelpad pushes the rotated ylabel left so it clears the leftmost
    # xtick label ("UI+Ad Pref") that extends down-left at 30 degrees.
    ax.set_ylabel(r"per-prompt $T$  (mean pairwise $\tau$)",
                  fontsize=12, labelpad=8)
    ax.tick_params(axis="y", labelsize=10)
    # Legend well below the rotated xtick labels.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.40),
              fontsize=10, ncol=4, framealpha=0.9, columnspacing=1.2,
              handlelength=2.0, borderaxespad=0.4)
    ax.set_ylim(-0.4, 1.28)

    # Cohort labels in the empty zone at the top of the plot frame, well
    # above the violin peaks (which top out near 1.0).
    ax.text(2.0, 1.18, "Aesthetics cohort", ha="center", va="center",
            fontsize=11, fontweight="bold", color=COHORT_COLOR["Aesthetics"])
    ax.text(6.5, 1.18, "Descriptions cohort", ha="center", va="center",
            fontsize=11, fontweight="bold", color=COHORT_COLOR["Descriptions"])

    fig.tight_layout()
    fig.savefig(OUT / "fig_cross_domain.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote: {OUT / 'fig_cross_domain.png'}")


# ---- Figure 2: per-dim cycle rates ----------------------------------------

def fig_cycle_rates(stats, refs, null_rate=0.2113):
    """Per-dim cycle rate bars with null line and reference cycle rates."""
    # Smaller figsize since this is displayed side-by-side at ~0.46\linewidth.
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    positions = np.arange(len(DIMS))
    rates = [float(stats[slug]["cycle"].mean()) for slug, _, _ in DIMS]
    colors = [COHORT_COLOR[g] for _, _, g in DIMS]
    bars = ax.bar(positions, rates, color=colors, edgecolor="black",
                  linewidth=0.5, alpha=0.85)
    for b, r in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.007,
                f"{r:.2f}", ha="center", va="bottom", fontsize=10)

    # null line (red dashed)
    ax.axhline(null_rate, color="red", linestyle="--", linewidth=1.0,
               label=f"iid-uniform null ({null_rate:.3f})")
    # reference dataset cycle rates
    refs_cycle = {
        "Sushi":      float(refs["sushi"]["cycle"].mean()),
        "MovieLens":  float(refs["movielens"]["cycle"].mean()),
        "HPSv2-test": float(refs["hpsv2"]["cycle"].mean()),
    }
    for name, v in refs_cycle.items():
        ax.axhline(v, color=REF_COLOR[name], linestyle=":", linewidth=1.0,
                   alpha=0.85, label=f"{name} ({v:.3f})")

    ax.axvline(4.5, color="black", linewidth=0.4, alpha=0.4)
    ax.set_xticks(positions)
    ax.set_xticklabels([d[1] for d in DIMS], rotation=30, ha="right",
                       fontsize=11)
    ax.set_ylabel("Condorcet cycle rate", fontsize=13)
    ax.set_ylim(0, max(0.28, max(rates) * 1.30))
    ax.tick_params(axis="y", labelsize=11)
    ax.legend(loc="upper right", fontsize=10, ncol=2, framealpha=0.92,
              handlelength=1.8, columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cycle_rates.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote: {OUT / 'fig_cycle_rates.png'}")


# ---- Figure 3: sub-dim signal ordering -----------------------------------

def fig_subdim_signal(stats, refs):
    """Mean pair-tau per dim, sorted descending, color-coded by cohort."""
    rows = [(slug, display, group, float(stats[slug]["pair_tau"].mean()))
            for slug, display, group in DIMS]
    rows.sort(key=lambda r: -r[3])

    # Smaller figsize since this is displayed side-by-side at ~0.46\linewidth.
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    positions = np.arange(len(rows))
    means = [r[3] for r in rows]
    colors = [COHORT_COLOR[r[2]] for r in rows]
    bars = ax.bar(positions, means, color=colors, edgecolor="black",
                  linewidth=0.5, alpha=0.85)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.006,
                f"{m:+.3f}", ha="center", va="bottom", fontsize=9)

    refs_pt = {
        "Sushi":      float(refs["sushi"]["pair_tau"].mean()),
        "MovieLens":  float(refs["movielens"]["pair_tau"].mean()),
        "HPSv2-test": float(refs["hpsv2"]["pair_tau"].mean()),
    }
    for name, v in refs_pt.items():
        ax.axhline(v, color=REF_COLOR[name], linestyle="--", linewidth=1.0,
                   alpha=0.85, label=f"{name} ({v:+.3f})")

    ax.set_xticks(positions)
    ax.set_xticklabels([r[1] for r in rows], rotation=30, ha="right",
                       fontsize=11)
    ax.set_ylabel(r"mean pairwise $\tau$", fontsize=13)
    ax.set_ylim(0, 0.42)
    ax.tick_params(axis="y", labelsize=11)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.92,
              handlelength=1.8)

    # cohort legend handles
    from matplotlib.patches import Patch
    cohort_handles = [Patch(facecolor=COHORT_COLOR["Aesthetics"],
                            edgecolor="black", label="Aesthetics"),
                      Patch(facecolor=COHORT_COLOR["Descriptions"],
                            edgecolor="black", label="Descriptions")]
    ax2 = ax.twinx()
    ax2.set_yticks([])
    ax2.legend(handles=cohort_handles, loc="upper left", fontsize=10,
               framealpha=0.92)

    fig.tight_layout()
    fig.savefig(OUT / "fig_subdim_signal.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote: {OUT / 'fig_subdim_signal.png'}")


def main():
    stats = load_dim_stats()
    refs = load_refs()
    fig_cross_domain(stats, refs)
    fig_cycle_rates(stats, refs)
    fig_subdim_signal(stats, refs)


if __name__ == "__main__":
    main()
