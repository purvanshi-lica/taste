"""
Per-evaluator "agreeableness" view: which reviewers consistently agree
with their peers, and which are outliers?

For each (dimension, evaluator), we collect every Kendall tau value
between that evaluator and the 4 partners they participate in, across
all 80 prompts.  That gives 320 tau values per cell.

Two views per cohort:

  1. **Mean heatmap + overall bar chart**
     `_evaluator_agreeableness_<cohort>.png`
     One number per cell: mean over the 320 values.

  2. **Histogram grid**
     `_evaluator_distribution_<cohort>.png`
     Each cell is a histogram of those 320 tau values overlaid on the
     R=2 Mahonian null.  Reveals the *shape* of agreement -- e.g. an
     evaluator whose mean is +0.15 because half their tau values are
     +0.33 and half are -0.33 looks very different from one whose mean
     is +0.15 because most of their tau values are +0.15.

Usage:
    python evaluator_agreeableness.py
"""

import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from taste_stats import DIMENSIONS, load_rank_tensor, compute_stats, DATA_DIR


def per_evaluator_tau_arrays(stats):
    """For each evaluator e, return concatenated tau values across all
    (partner, prompt) pairs they participate in.  Returns dict {e: array}."""
    pair_labels = stats["pair_tau_labels"]
    pair_tau = stats["pair_tau"]                      # (n_prompts, 10)
    by_eval = {}
    for col, (e1, e2) in enumerate(pair_labels):
        by_eval.setdefault(e1, []).append(pair_tau[:, col])
        by_eval.setdefault(e2, []).append(pair_tau[:, col])
    return {e: np.concatenate(vs) for e, vs in by_eval.items()}


def per_evaluator_mean_tau(stats):
    """Mean version of `per_evaluator_tau_arrays`."""
    return {e: float(arr.mean()) for e, arr in per_evaluator_tau_arrays(stats).items()}


def build_cohort_heatmap(group_name, group_dims, out_path,
                          out_path_dist=None):
    print(f"\n=== {group_name} ===")
    matrix = {}                                          # mean tau per (e, d)
    arrays = {}                                          # full tau arrays per (e, d)
    dim_displays = []
    eval_order = None
    for slug, display, group, csv_name in group_dims:
        tensor, _, evaluators = load_rank_tensor(str(DATA_DIR / csv_name))
        stats = compute_stats(tensor, evaluators)
        per_eval_arr = per_evaluator_tau_arrays(stats)
        per_eval_mean = {e: float(a.mean()) for e, a in per_eval_arr.items()}
        if eval_order is None:
            eval_order = sorted(per_eval_mean.keys())
        dim_displays.append(display)
        for e in eval_order:
            matrix.setdefault(e, []).append(per_eval_mean[e])
            arrays[(e, display)] = per_eval_arr[e]

    mat = np.array([matrix[e] for e in eval_order])    # (n_eval, n_dim)

    print("  Overall agreeableness ranking (mean across dims):")
    overall = mat.mean(axis=1)
    order = np.argsort(-overall)
    for i in order:
        print(f"    {eval_order[i]:25s}  mean tau across dims = {overall[i]:+.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4 + 0.4 * len(eval_order)),
                              gridspec_kw={"width_ratios": [1.6, 1]})

    # Heatmap: evaluators x dimensions
    sns.heatmap(mat, ax=axes[0], annot=True, fmt="+.3f", cmap="RdYlGn",
                vmin=-0.1, vmax=0.4, cbar_kws={"label": r"mean pairwise $\tau$"},
                xticklabels=dim_displays, yticklabels=eval_order)
    axes[0].set_title(f"{group_name}: per-evaluator mean pairwise $\\tau$ "
                      f"vs partners (averaged across 80 prompts)",
                      fontsize=11)
    axes[0].tick_params(axis="x", rotation=25, labelsize=9)
    axes[0].tick_params(axis="y", rotation=0, labelsize=9)

    # Bar chart: overall agreeableness, sorted desc
    ax = axes[1]
    bars = ax.barh(np.arange(len(eval_order))[::-1],
                   [overall[i] for i in order],
                   color=plt.cm.RdYlGn((overall[order] + 0.1) / 0.5),
                   edgecolor="black")
    ax.set_yticks(np.arange(len(eval_order))[::-1])
    ax.set_yticklabels([eval_order[i] for i in order], fontsize=9)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("mean $\\tau$ across dims and partners", fontsize=10)
    ax.set_title(f"Overall agreeableness — {group_name}",
                 fontsize=11)
    for j, b in enumerate(bars):
        v = overall[order[j]]
        ax.text(v + 0.005, b.get_y() + b.get_height() / 2,
                f"{v:+.3f}", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote: {out_path}")

    # ---- Histogram grid: one histogram per (evaluator, dim) cell ----
    if out_path_dist is not None:
        from null_kendall import r2_tau_pmf
        pmf = r2_tau_pmf(4)
        null_taus = np.array(sorted(float(t) for t in pmf.keys()))
        null_ps = np.array([float(pmf[t]) for t in sorted(pmf.keys(), key=float)])
        width = 0.8 * float(np.median(np.diff(null_taus)))

        # Sort evaluators by overall agreeableness descending so the most
        # agreeable evaluator is the top row.
        eval_sorted = [eval_order[i] for i in order]

        n_eval = len(eval_sorted)
        n_dim = len(dim_displays)
        fig, axes = plt.subplots(n_eval, n_dim,
                                  figsize=(2.6 * n_dim, 1.9 * n_eval),
                                  sharex=True, sharey=True)
        if n_eval == 1:
            axes = axes[None, :]
        if n_dim == 1:
            axes = axes[:, None]
        for ri, e in enumerate(eval_sorted):
            for ci, d in enumerate(dim_displays):
                ax = axes[ri, ci]
                vals = arrays[(e, d)]
                idx = np.argmin(np.abs(vals[:, None] - null_taus[None, :]),
                                 axis=1)
                counts = np.bincount(idx, minlength=len(null_taus))
                probs = counts / counts.sum()
                ax.bar(null_taus, null_ps, width=width, color="lightgray",
                       edgecolor="gray", linewidth=0.4, alpha=0.85)
                ax.bar(null_taus, probs, width=width * 0.45,
                       color="black", edgecolor="black", linewidth=0.4,
                       alpha=0.95)
                ax.axvline(0, color="gray", lw=0.3, ls=":")
                ax.text(-1.0, 0.32, f"mean={vals.mean():+.2f}",
                        fontsize=7, va="top", ha="left")
                if ri == 0:
                    ax.set_title(d, fontsize=9)
                if ci == 0:
                    ax.set_ylabel(e[:14], fontsize=9, rotation=0,
                                  ha="right", va="center", labelpad=30)
                if ri == n_eval - 1:
                    ax.set_xlabel(r"$\tau$", fontsize=8)
                ax.set_xlim(-1.1, 1.1)
                ax.set_ylim(0, 0.4)
                ax.tick_params(axis="both", labelsize=7)
        fig.suptitle(f"{group_name}: per-evaluator pairwise $\\tau$ distribution "
                     f"  (rows = evaluators sorted by overall mean, cols = dims; "
                     f"each cell is a histogram of 4 partners x 80 prompts = 320 tau values "
                     f"on the Mahonian null in grey)",
                     fontsize=11, y=1.02)
        fig.tight_layout()
        fig.savefig(out_path_dist, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Wrote: {out_path_dist}")

    return eval_order, dim_displays, mat, overall


def build_cohort_pair_views(group_name, group_dims, out_dist_path,
                              out_heatmap_path):
    """Cohort-aggregated per-pair distribution grid + 5x5 mean-tau heatmap.

    The 10 evaluator pairs each produce ~80 * n_dims tau values when
    pooled across the cohort.  Histogram each pair's pooled distribution,
    sorted by mean tau descending so groupings + outliers stand out.
    Also build the symmetric 5x5 mean-tau heatmap.
    """
    from null_kendall import r2_tau_pmf
    print(f"\n=== {group_name} (per-pair cohort view) ===")
    pair_arrays = {}                    # {(e1, e2): np.concat across dims}
    eval_set = set()
    for slug, display, group, csv_name in group_dims:
        tensor, _, evaluators = load_rank_tensor(str(DATA_DIR / csv_name))
        stats = compute_stats(tensor, evaluators)
        labels = stats["pair_tau_labels"]
        pair_tau = stats["pair_tau"]
        for col, (e1, e2) in enumerate(labels):
            key = tuple(sorted([e1, e2]))
            pair_arrays.setdefault(key, []).append(pair_tau[:, col])
            eval_set.update([e1, e2])
    pair_arrays = {k: np.concatenate(v) for k, v in pair_arrays.items()}
    eval_order = sorted(eval_set)

    # 5x5 heatmap of mean tau per pair
    n = len(eval_order)
    mat = np.full((n, n), np.nan)
    np.fill_diagonal(mat, 1.0)
    for (e1, e2), arr in pair_arrays.items():
        i, j = eval_order.index(e1), eval_order.index(e2)
        mat[i, j] = mat[j, i] = float(arr.mean())

    fig, ax = plt.subplots(figsize=(8, 6))
    short = [e.split()[0][:10] for e in eval_order]
    sns.heatmap(mat, ax=ax, annot=True, fmt="+.3f", cmap="RdYlGn",
                vmin=-0.05, vmax=0.4,
                xticklabels=short, yticklabels=short, square=True,
                cbar_kws={"label": r"mean pairwise $\tau$ across dims"})
    ax.set_title(f"{group_name}: cohort-level mean pairwise $\\tau$ "
                 f"(across all dims and 80 prompts)",
                 fontsize=11)
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    fig.tight_layout()
    fig.savefig(out_heatmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote: {out_heatmap_path}")

    # 10-pair distribution grid
    pmf = r2_tau_pmf(4)
    null_taus = np.array(sorted(float(t) for t in pmf.keys()))
    null_ps = np.array([float(pmf[t]) for t in sorted(pmf.keys(), key=float)])
    width = 0.8 * float(np.median(np.diff(null_taus)))

    pairs_sorted = sorted(pair_arrays.items(),
                          key=lambda kv: -kv[1].mean())
    n_pairs = len(pairs_sorted)
    rows, cols = 2, 5
    fig, axes = plt.subplots(rows, cols, figsize=(15, 6),
                              sharex=True, sharey=True)
    for k, ((e1, e2), arr) in enumerate(pairs_sorted):
        ax = axes[k // cols, k % cols]
        idx = np.argmin(np.abs(arr[:, None] - null_taus[None, :]), axis=1)
        counts = np.bincount(idx, minlength=len(null_taus))
        probs = counts / counts.sum()
        ax.bar(null_taus, null_ps, width=width, color="lightgray",
               edgecolor="gray", linewidth=0.4, alpha=0.85,
               label="Mahonian null")
        ax.bar(null_taus, probs, width=width * 0.45, color="black",
               edgecolor="black", linewidth=0.4, alpha=0.95)
        ax.axvline(0, color="gray", lw=0.4, ls=":")
        e1s = e1.split()[0][:8]
        e2s = e2.split()[0][:8]
        ax.set_title(f"{e1s} x {e2s}\nmean={arr.mean():+.3f},  "
                     f"median={float(np.median(arr)):+.3f}",
                     fontsize=9)
        if k // cols == rows - 1:
            ax.set_xlabel(r"$\tau$", fontsize=9)
        if k % cols == 0:
            ax.set_ylabel("prob", fontsize=9)
        ax.set_xlim(-1.1, 1.1)

    fig.suptitle(f"{group_name}: pairwise $\\tau$ distribution per evaluator "
                 f"pair, pooled across all dims and 80 prompts "
                 f"(top-left = most-agreeing pair, bottom-right = least-agreeing)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote: {out_dist_path}")
    return pair_arrays, eval_order, mat


def main():
    out_dir = HERE / "figures"
    out_dir.mkdir(exist_ok=True)
    aes_dims = [(s, d, g, c) for s, d, g, c in DIMENSIONS if g == "Aesthetics"]
    desc_dims = [(s, d, g, c) for s, d, g, c in DIMENSIONS if g == "Descriptions"]

    aes = build_cohort_heatmap(
        "Aesthetics cohort", aes_dims,
        out_dir / "_evaluator_agreeableness_aesthetics.png",
        out_path_dist=out_dir / "_evaluator_distribution_aesthetics.png",
    )
    desc = build_cohort_heatmap(
        "Descriptions cohort", desc_dims,
        out_dir / "_evaluator_agreeableness_descriptions.png",
        out_path_dist=out_dir / "_evaluator_distribution_descriptions.png",
    )

    # Cohort-level per-pair views (10 pairs each, pooled across dims)
    build_cohort_pair_views(
        "Aesthetics cohort", aes_dims,
        out_dir / "_pair_distribution_aesthetics.png",
        out_dir / "_pair_heatmap_aesthetics.png",
    )
    build_cohort_pair_views(
        "Descriptions cohort", desc_dims,
        out_dir / "_pair_distribution_descriptions.png",
        out_dir / "_pair_heatmap_descriptions.png",
    )

    # Save summary text
    summary = ["# Per-evaluator agreeableness", ""]
    for label, (eval_order, dim_displays, mat, overall) in [
        ("Aesthetics", aes), ("Descriptions", desc),
    ]:
        order = np.argsort(-overall)
        summary.append(f"### {label} cohort  (5 evaluators across "
                       f"{mat.shape[1]} sub-dimensions, 80 prompts each)")
        summary.append("")
        summary.append("| Rank | Evaluator | overall mean $\\tau$ | "
                       + " | ".join(dim_displays) + " |")
        summary.append("|---|---|---|" + "---|" * len(dim_displays))
        for rk, i in enumerate(order, 1):
            row = (f"| {rk} | {eval_order[i]} | **{overall[i]:+.3f}** |  "
                   + "  |  ".join(f"{v:+.3f}" for v in mat[i]) + " |")
            summary.append(row)
        summary.append("")
        summary.append(f"**Most agreeable:** {eval_order[order[0]]}  "
                       f"({overall[order[0]]:+.3f}).  "
                       f"**Least agreeable:** {eval_order[order[-1]]}  "
                       f"({overall[order[-1]]:+.3f}).  "
                       f"Spread = {overall[order[0]] - overall[order[-1]]:+.3f}.")
        summary.append("")

    with open(HERE / "figures" / "_evaluator_agreeableness.md", "w") as f:
        f.write("\n".join(summary))
    print("\nSummary table written to figures/_evaluator_agreeableness.md")


if __name__ == "__main__":
    main()
