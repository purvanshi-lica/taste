"""
Hallucination flag inter-rater agreement (binary).

Input: `hallucination_flags_*.csv` (Round 1, aesthetics cohort) and
`v2_hallucination_flags_*.csv` (Round 2, descriptions cohort).  Each file
has one row per (prompt_id, model, evaluator) with a 3-way flag
(``No Hallucination`` / ``Minor`` / ``Major``).  We binarise the flag in
one of two ways depending on ``--binarisation``:

* ``minor_or_major``  (default): ``hallucinated = (value > 0)``.  Merges
  Minor + Major into one positive class (~45% prevalence).
* ``major_only``: ``hallucinated = (value == 2)``.  Only Major counts;
  Minor is pooled with No Hallucination (~10% prevalence).  This asks a
  stricter question: do evaluators agree on what is a *severe* failure?

For each round + overall we compute:
  - Flag prevalence (overall, per model, per evaluator)
  - Fleiss's kappa (5 raters on 320 binary items)
  - Krippendorff's alpha (nominal)
  - Prevalence- and bias-adjusted kappa (PABAK) -- useful when prevalence
    is far from 50/50, as in the ``major_only`` case.
  - Positive specific agreement -- fraction of positive votes that
    co-occur between evaluator pairs; informative for rare positives.
  - Pairwise Cohen's kappa heatmap (5x5 per round)
  - Vote-count histogram: of the 320 items, how many got 0/1/2/3/4/5 "hallucinated" votes?
  - Unanimous-consensus rate (all 5 agree) and strong-majority rate (>=4/5)

Writes figures to ``analysis_outputs/hallucination/<mode>/`` and a report to
``HALLUCINATION_AGREEMENT_REPORT_<mode>.md`` in the data_analysis dir.

Usage
-----
    # Default: Minor + Major treated as hallucinated (~45% prevalence)
    python hallucination_agreement.py

    # Strict: only Major counts as hallucinated (~10% prevalence)
    python hallucination_agreement.py --binarisation major_only
"""

import argparse
import os
from itertools import combinations

import krippendorff
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(SCRIPT_DIR, "csv")

ROUND_FILES = {
    "round1_aesthetics":  "hallucination_flags_2026-04-24T0822.csv",
    "round2_descriptions": "v2_hallucination_flags_2026-04-24T0823.csv",
}

MODELS = ["FLUX.2 [max]", "GPT Image 1.5", "Nano Banana 2", "Seedream 5.0 Lite"]
MODEL_SHORT = {"FLUX.2 [max]": "FLUX.2", "GPT Image 1.5": "GPT-Img",
               "Nano Banana 2": "NanoBan", "Seedream 5.0 Lite": "Seedream"}

BINARISATIONS = {
    "minor_or_major": {
        "rule": lambda v: int(v > 0),
        "positive_label": "Minor + Major",
        "description": "Minor and Major collapsed to 1; No Hallucination = 0.",
    },
    "major_only": {
        "rule": lambda v: int(v == 2),
        "positive_label": "Major only",
        "description": "Major = 1; No Hallucination and Minor both map to 0.",
    },
}


def cohen_kappa(y1, y2):
    """Cohen's kappa for two binary raters on aligned arrays."""
    y1, y2 = np.asarray(y1), np.asarray(y2)
    n = len(y1)
    po = float((y1 == y2).mean())
    p1_y1, p1_y2 = y1.mean(), y2.mean()
    pe = p1_y1 * p1_y2 + (1 - p1_y1) * (1 - p1_y2)
    return (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0, po


def pabak(y1, y2):
    """Prevalence- and bias-adjusted kappa (Byrt et al. 1993).

    When class prevalence is skewed (~10% positive in our ``major_only``
    case), raw Cohen's kappa can be misleadingly low even at high percent
    agreement because expected chance-agreement is inflated.  PABAK rescales
    to ``2 * po - 1``, ignoring the marginal class imbalance.  It equals
    Cohen's kappa when prevalence is exactly 50/50 and differs otherwise.
    """
    po = float((np.asarray(y1) == np.asarray(y2)).mean())
    return 2 * po - 1


def positive_specific_agreement(y1, y2):
    """Dice / F1-style agreement on positive-only calls.

    ``2 * sum(y1 AND y2) / (sum(y1) + sum(y2))``.  Answers "when raters do
    call an item positive, how often do they agree?" -- uses only positive
    votes, so it is robust when negatives dominate.  Returns np.nan if
    neither rater ever votes positive.
    """
    y1, y2 = np.asarray(y1), np.asarray(y2)
    denom = y1.sum() + y2.sum()
    if denom == 0:
        return float("nan")
    return float(2 * (y1 & y2).sum() / denom)


def fleiss_kappa(matrix):
    """Fleiss's kappa for a (n_items x n_categories) count matrix.

    Each row sums to n_raters (5 here).  Returns kappa in [-1, 1].
    """
    matrix = np.asarray(matrix, dtype=float)
    n_items, n_cat = matrix.shape
    n_raters = matrix.sum(axis=1)[0]
    # P_i: per-item agreement
    P_i = ((matrix ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = P_i.mean()
    # P_e: expected agreement by chance
    p_j = matrix.sum(axis=0) / (n_items * n_raters)
    P_e = (p_j ** 2).sum()
    return (P_bar - P_e) / (1 - P_e) if (1 - P_e) > 0 else 0.0


def verdict_kappa(k):
    """Landis & Koch (1977) magnitude guidelines."""
    if k < 0:
        return "worse than chance"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


def load_and_binarize(csv_file, rule):
    df = pd.read_csv(os.path.join(CSV_DIR, csv_file))
    df["hallucinated"] = df["hallucination_value"].apply(rule).astype(int)
    dup_mask = df.duplicated(subset=["prompt_id", "model", "evaluator"], keep=False)
    if dup_mask.any():
        print(f"  WARN: {dup_mask.sum()} duplicate (prompt,model,eval) rows in {csv_file}")
    return df


def analyze_round(round_slug, df):
    """Return a dict of all agreement metrics for one round."""
    evaluators = sorted(df["evaluator"].unique())
    n_ev = len(evaluators)

    # Item-level pivot: rows = (prompt_id, model), cols = evaluator, value = 0/1
    pivot = df.pivot_table(
        values="hallucinated", index=["prompt_id", "model"],
        columns="evaluator", aggfunc="first",
    )[evaluators]

    # Drop any item missing any evaluator (shouldn't happen with 5x5 design)
    pivot = pivot.dropna()
    pivot = pivot.astype(int)
    n_items = len(pivot)

    # Vote-count histogram: for each item, how many of 5 evaluators flagged it?
    vote_counts = pivot.sum(axis=1).value_counts().sort_index()
    # Ensure all 0..5 buckets appear
    vote_counts = vote_counts.reindex(range(n_ev + 1), fill_value=0)

    unanimous = int(vote_counts.iloc[0]) + int(vote_counts.iloc[-1])
    # "Strong majority": items where ≥4 of 5 evaluators agree (i.e. vote count
    # ≤1 for "no" consensus or ≥4 for "yes" consensus).  Explicitly exclude
    # the ambiguous middle (2/5, 3/5).
    votes = pivot.sum(axis=1).values
    majority = int(((votes <= 1) | (votes >= n_ev - 1)).sum())

    # Fleiss's kappa on a (n_items x 2) count matrix
    count_mat = np.zeros((n_items, 2), dtype=int)
    count_mat[:, 1] = pivot.sum(axis=1).values
    count_mat[:, 0] = n_ev - count_mat[:, 1]
    fk = fleiss_kappa(count_mat)

    # Krippendorff's alpha (nominal)
    reliability = pivot.values.T.astype(float)  # (n_raters x n_items)
    ka = krippendorff.alpha(reliability_data=reliability, level_of_measurement="nominal")

    # Per-pair Cohen's kappa, PABAK, and positive-specific agreement
    pair_kappa = np.full((n_ev, n_ev), np.nan)
    pair_po = np.full((n_ev, n_ev), np.nan)
    pair_pabak = np.full((n_ev, n_ev), np.nan)
    pair_psa = np.full((n_ev, n_ev), np.nan)
    np.fill_diagonal(pair_kappa, 1.0)
    np.fill_diagonal(pair_pabak, 1.0)
    np.fill_diagonal(pair_psa, 1.0)
    for i, j in combinations(range(n_ev), 2):
        y1 = pivot.iloc[:, i].values
        y2 = pivot.iloc[:, j].values
        k, po = cohen_kappa(y1, y2)
        pair_kappa[i, j] = pair_kappa[j, i] = k
        pair_po[i, j] = pair_po[j, i] = po
        pair_pabak[i, j] = pair_pabak[j, i] = pabak(y1, y2)
        pair_psa[i, j] = pair_psa[j, i] = positive_specific_agreement(y1, y2)
    # Aggregate over off-diagonal pairs
    mask = ~np.eye(n_ev, dtype=bool)
    mean_kappa = float(np.nanmean(pair_kappa[mask]))
    mean_pabak = float(np.nanmean(pair_pabak[mask]))
    mean_psa = float(np.nanmean(pair_psa[mask]))
    mean_po = float(np.nanmean(pair_po[mask]))

    # Flag prevalence
    overall_rate = float(df["hallucinated"].mean())
    per_model = df.groupby("model")["hallucinated"].mean().reindex(MODELS)
    per_eval = df.groupby("evaluator")["hallucinated"].mean().reindex(evaluators)

    # 3-way category rates (informational)
    three_way = df["hallucination_flag"].value_counts(normalize=True)

    return {
        "round_slug": round_slug,
        "n_items": n_items,
        "evaluators": evaluators,
        "fleiss_kappa": fk,
        "krippendorff_alpha": ka,
        "overall_rate": overall_rate,
        "per_model": per_model,
        "per_evaluator": per_eval,
        "three_way": three_way,
        "vote_counts": vote_counts,
        "pair_kappa": pair_kappa,
        "pair_po": pair_po,
        "pair_pabak": pair_pabak,
        "pair_psa": pair_psa,
        "mean_kappa": mean_kappa,
        "mean_pabak": mean_pabak,
        "mean_psa": mean_psa,
        "mean_po": mean_po,
        "unanimous_items": unanimous,
        "majority_items": majority,
        "pivot": pivot,
    }


def build_combined(per_round):
    """Concatenate pivots and re-compute Fleiss's + Krippendorff's across both rounds.

    NOTE: evaluator sets differ per round, so the pooled metric treats the
    two rounds as independent item sets using the union of evaluators; we
    therefore report Fleiss on each round separately and simply combine the
    vote-count histograms for an "overall" picture of consensus strength.
    The combined Fleiss here is a *within-item* metric -- fine because each
    item still has exactly 5 votes, just from different evaluator pools.
    """
    frames = []
    for r in per_round:
        p = r["pivot"].copy()
        p.columns = [f"{r['round_slug']}:{c}" for c in p.columns]
        p["n_hallucinated"] = p.sum(axis=1)
        frames.append(p[["n_hallucinated"]])
    comb = pd.concat(frames, axis=0)
    n_items = len(comb)
    n_ev = 5

    count_mat = np.zeros((n_items, 2), dtype=int)
    count_mat[:, 1] = comb["n_hallucinated"].values
    count_mat[:, 0] = n_ev - count_mat[:, 1]
    fk = fleiss_kappa(count_mat)

    vote_counts = comb["n_hallucinated"].value_counts().sort_index()
    vote_counts = vote_counts.reindex(range(n_ev + 1), fill_value=0)

    unanimous = int(vote_counts.iloc[0]) + int(vote_counts.iloc[-1])
    return {
        "n_items": n_items,
        "fleiss_kappa": fk,
        "vote_counts": vote_counts,
        "unanimous_items": unanimous,
    }


def save_pair_kappa_heatmap(per_round, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, r in zip(axes, per_round):
        short_names = [e[:12] for e in r["evaluators"]]
        sns.heatmap(r["pair_kappa"], ax=ax, annot=True, fmt=".2f", cmap="RdYlGn",
                    vmin=-0.2, vmax=1.0,
                    xticklabels=short_names, yticklabels=short_names,
                    square=True, cbar_kws={"shrink": 0.8})
        ax.set_title(f"{r['round_slug']}\nFleiss kappa = {r['fleiss_kappa']:.3f}", fontsize=12)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
    fig.suptitle("Pairwise Cohen's kappa — hallucinated (binary)", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_vote_histogram(per_round, combined, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, r in zip(axes[:2], per_round):
        counts = r["vote_counts"].values
        n_ev = len(counts) - 1
        x = np.arange(n_ev + 1)
        colors = ["#388e3c" if c in (0, n_ev) else "#f57c00" for c in x]
        ax.bar(x, counts, color=colors, edgecolor="black", alpha=0.85)
        for xi, yi in zip(x, counts):
            if yi > 0:
                ax.text(xi, yi + 1, str(int(yi)), ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xlabel("# evaluators voting 'hallucinated'", fontsize=11)
        ax.set_ylabel("# items (prompt × model)" if ax is axes[0] else "")
        ax.set_title(
            f"{r['round_slug']}\n"
            f"unanimous {r['unanimous_items']}/{r['n_items']} "
            f"({100*r['unanimous_items']/r['n_items']:.1f}%)",
            fontsize=12,
        )

    counts = combined["vote_counts"].values
    n_ev = len(counts) - 1
    x = np.arange(n_ev + 1)
    colors = ["#388e3c" if c in (0, n_ev) else "#f57c00" for c in x]
    axes[2].bar(x, counts, color=colors, edgecolor="black", alpha=0.85)
    for xi, yi in zip(x, counts):
        if yi > 0:
            axes[2].text(xi, yi + 1, str(int(yi)), ha="center", va="bottom", fontsize=9)
    axes[2].set_xticks(x)
    axes[2].set_xlabel("# evaluators voting 'hallucinated'", fontsize=11)
    axes[2].set_title(
        f"Combined (both rounds)\n"
        f"unanimous {combined['unanimous_items']}/{combined['n_items']} "
        f"({100*combined['unanimous_items']/combined['n_items']:.1f}%), "
        f"Fleiss kappa = {combined['fleiss_kappa']:.3f}",
        fontsize=12,
    )

    fig.suptitle("Distribution of 'hallucinated' vote counts across items",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_rate_by_model_and_evaluator(per_round, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for col, r in enumerate(per_round):
        # Top row: per-model rates
        ax = axes[0, col]
        per_model = r["per_model"]
        short = [MODEL_SHORT[m] for m in per_model.index]
        colors = plt.cm.Set2(np.linspace(0, 1, len(short)))
        bars = ax.bar(short, per_model.values, color=colors, edgecolor="black", alpha=0.85)
        for b, v in zip(bars, per_model.values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Hallucination rate" if col == 0 else "")
        ax.set_title(f"{r['round_slug']} — by model", fontsize=12)
        ax.axhline(r["overall_rate"], linestyle="--", color="gray", linewidth=1,
                   label=f"Overall = {r['overall_rate']:.2f}")
        ax.legend(fontsize=9)

        # Bottom row: per-evaluator rates
        ax = axes[1, col]
        per_eval = r["per_evaluator"]
        short_ev = [e[:12] for e in per_eval.index]
        bars = ax.bar(short_ev, per_eval.values,
                      color="#1976d2", edgecolor="black", alpha=0.85)
        for b, v in zip(bars, per_eval.values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Hallucination rate" if col == 0 else "")
        ax.set_title(f"{r['round_slug']} — by evaluator", fontsize=12)
        ax.axhline(r["overall_rate"], linestyle="--", color="gray", linewidth=1)
        ax.tick_params(axis="x", rotation=25)

    fig.suptitle("Hallucination flag rate (binary: Minor+Major vs None)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def write_report(per_round, combined, out_path, binarisation):
    info = BINARISATIONS[binarisation]
    lines = []
    lines.append(f"# Hallucination Flag Agreement — binarisation: `{binarisation}`")
    lines.append("")
    lines.append(f"**Positive class:** {info['positive_label']} — {info['description']}")
    lines.append("")
    lines.append("Data: 80 prompts × 4 models = **320 items per round**, rated "
                 "by 5 evaluators each.  Two rounds with disjoint evaluator "
                 "pools (aesthetics cohort vs descriptions cohort).")
    lines.append("")
    lines.append("---")
    lines.append("")

    # TL;DR
    r1, r2 = per_round
    def _rate_range(r):
        rates = r["per_evaluator"].values
        lo_ev = r["per_evaluator"].idxmin()
        hi_ev = r["per_evaluator"].idxmax()
        return rates.min(), rates.max(), lo_ev, hi_ev

    lo1, hi1, lo_ev1, hi_ev1 = _rate_range(r1)
    lo2, hi2, lo_ev2, hi_ev2 = _rate_range(r2)

    lines.append("## TL;DR")
    lines.append("")
    lines.append(
        f"Binarisation: **{binarisation}** — {BINARISATIONS[binarisation]['description']}  "
        f"Positive prevalence: **{r1['overall_rate']:.1%}** (R1) / "
        f"**{r2['overall_rate']:.1%}** (R2)."
    )
    lines.append("")
    lines.append(
        f"- **Fleiss kappa:** {r1['fleiss_kappa']:.3f} / "
        f"{r2['fleiss_kappa']:.3f} / combined {combined['fleiss_kappa']:.3f}  "
        f"— *{verdict_kappa(combined['fleiss_kappa'])}* on Landis-Koch."
    )
    lines.append(
        f"- **Krippendorff alpha (nominal):** "
        f"{r1['krippendorff_alpha']:.3f} / {r2['krippendorff_alpha']:.3f}"
    )
    lines.append(
        f"- **Mean Cohen kappa across all pairs:** "
        f"{r1['mean_kappa']:.3f} / {r2['mean_kappa']:.3f}"
    )
    lines.append(
        f"- **PABAK (prevalence-adjusted, important for skewed classes):** "
        f"{r1['mean_pabak']:.3f} / {r2['mean_pabak']:.3f}"
    )
    lines.append(
        f"- **Positive specific agreement (Dice over positive votes):** "
        f"{r1['mean_psa']:.3f} / {r2['mean_psa']:.3f}"
    )
    lines.append(
        f"- **Raw percent agreement (averaged over pairs):** "
        f"{100*r1['mean_po']:.1f}% / {100*r2['mean_po']:.1f}%"
    )
    lines.append(
        f"- **Unanimous items (0/5 or 5/5):** "
        f"{r1['unanimous_items']}/{r1['n_items']} "
        f"({100*r1['unanimous_items']/r1['n_items']:.1f}%) / "
        f"{r2['unanimous_items']}/{r2['n_items']} "
        f"({100*r2['unanimous_items']/r2['n_items']:.1f}%) "
        f"/ combined {combined['unanimous_items']}/{combined['n_items']} "
        f"({100*combined['unanimous_items']/combined['n_items']:.1f}%)"
    )
    lines.append(
        f"- **Strong-majority items (≥4/5 agree):** "
        f"{r1['majority_items']}/{r1['n_items']} "
        f"({100*r1['majority_items']/r1['n_items']:.1f}%) / "
        f"{r2['majority_items']}/{r2['n_items']} "
        f"({100*r2['majority_items']/r2['n_items']:.1f}%)"
    )
    lines.append("")
    lines.append(
        f"**Per-evaluator flag rate spread** — a proxy for threshold "
        f"heterogeneity (same spread = everyone rates the same, big spread = "
        f"calibration drift):"
    )
    if lo1 > 0:
        lines.append(f"- Round 1: {100*lo1:.1f}% ({lo_ev1}) → {100*hi1:.1f}% "
                     f"({hi_ev1}), **{hi1/lo1:.1f}x spread**.")
    else:
        lines.append(f"- Round 1: {100*lo1:.1f}% ({lo_ev1}) → {100*hi1:.1f}% "
                     f"({hi_ev1}).")
    if lo2 > 0:
        lines.append(f"- Round 2: {100*lo2:.1f}% ({lo_ev2}) → {100*hi2:.1f}% "
                     f"({hi_ev2}), **{hi2/lo2:.1f}x spread**.")
    else:
        lines.append(f"- Round 2: {100*lo2:.1f}% ({lo_ev2}) → {100*hi2:.1f}% "
                     f"({hi_ev2}).")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Per-round agreement metrics")
    lines.append("")
    lines.append("| Metric | Round 1 (aesthetics, 5 evaluators) | Round 2 (descriptions, 5 evaluators) |")
    lines.append("|---|---|---|")
    lines.append(f"| Items (prompt × model) | {r1['n_items']} | {r2['n_items']} |")
    lines.append(f"| Fleiss's kappa (binary) | **{r1['fleiss_kappa']:.3f}** "
                 f"(*{verdict_kappa(r1['fleiss_kappa'])}*) | "
                 f"**{r2['fleiss_kappa']:.3f}** (*{verdict_kappa(r2['fleiss_kappa'])}*) |")
    lines.append(f"| Krippendorff's alpha (nominal) | **{r1['krippendorff_alpha']:.3f}** | "
                 f"**{r2['krippendorff_alpha']:.3f}** |")
    lines.append(f"| Overall flag rate | {r1['overall_rate']:.1%} | {r2['overall_rate']:.1%} |")
    lines.append(f"| Unanimous-consensus items (0/5 or 5/5) | "
                 f"{r1['unanimous_items']}/{r1['n_items']} ({100*r1['unanimous_items']/r1['n_items']:.1f}%) | "
                 f"{r2['unanimous_items']}/{r2['n_items']} ({100*r2['unanimous_items']/r2['n_items']:.1f}%) |")
    lines.append(f"| Strong-majority items (≥4/5 agree either way) | "
                 f"{r1['majority_items']}/{r1['n_items']} ({100*r1['majority_items']/r1['n_items']:.1f}%) | "
                 f"{r2['majority_items']}/{r2['n_items']} ({100*r2['majority_items']/r2['n_items']:.1f}%) |")
    lines.append("")
    lines.append("*Landis-Koch magnitude: <0.20 slight, 0.20-0.40 fair, "
                 "0.40-0.60 moderate, 0.60-0.80 substantial, >0.80 almost perfect.*")
    lines.append("")

    lines.append("### Vote-count distribution (out of 5 evaluators)")
    lines.append("")
    lines.append("| # voting 'hallucinated' | Round 1 | Round 2 | Combined |")
    lines.append("|---|---|---|---|")
    for k in range(6):
        v1 = r1["vote_counts"].iloc[k]
        v2 = r2["vote_counts"].iloc[k]
        vc = combined["vote_counts"].iloc[k]
        lines.append(f"| {k}/5 | {v1} | {v2} | {vc} |")
    lines.append("")

    lines.append("## Flag rate by model")
    lines.append("")
    lines.append("| Model | Round 1 | Round 2 |")
    lines.append("|---|---|---|")
    for m in MODELS:
        r1v = r1["per_model"].get(m, np.nan)
        r2v = r2["per_model"].get(m, np.nan)
        lines.append(f"| {MODEL_SHORT[m]} | {r1v:.2%} | {r2v:.2%} |")
    lines.append("")

    lines.append("## Flag rate by evaluator")
    lines.append("")
    for r in per_round:
        lines.append(f"**{r['round_slug']}:**")
        lines.append("")
        lines.append("| Evaluator | Flag rate |")
        lines.append("|---|---|")
        for ev, v in r["per_evaluator"].items():
            lines.append(f"| {ev} | {v:.2%} |")
        lines.append("")

    lines.append("## Pairwise Cohen's kappa")
    lines.append("")
    for r in per_round:
        lines.append(f"**{r['round_slug']}:**")
        lines.append("")
        pk = pd.DataFrame(r["pair_kappa"], index=r["evaluators"], columns=r["evaluators"])
        # Extract lower-triangle pair values
        pairs = []
        for i, j in combinations(range(len(r["evaluators"])), 2):
            e1, e2 = r["evaluators"][i], r["evaluators"][j]
            pairs.append((e1, e2, r["pair_kappa"][i, j]))
        pairs.sort(key=lambda x: x[2])
        lines.append("| Pair | Cohen's kappa | Magnitude |")
        lines.append("|---|---|---|")
        for e1, e2, k in pairs:
            lines.append(f"| {e1} × {e2} | {k:.3f} | *{verdict_kappa(k)}* |")
        lines.append("")

    lines.append("## Figures")
    lines.append("")
    lines.append("- `analysis_outputs/hallucination/01_pair_cohen_kappa.png` — 5×5 heatmap per round")
    lines.append("- `analysis_outputs/hallucination/02_vote_count_histogram.png` — distribution of "
                 "per-item vote counts (0..5 evaluators flagging)")
    lines.append("- `analysis_outputs/hallucination/03_rates_by_model_and_evaluator.png` — "
                 "hallucination rate per model + per evaluator")
    lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote: {out_path}")


def print_console_summary(per_round, combined):
    print("\n" + "=" * 70)
    print("HALLUCINATION FLAG BINARY AGREEMENT SUMMARY")
    print("=" * 70)
    for r in per_round:
        print(f"\n{r['round_slug']}:")
        print(f"  items: {r['n_items']}  overall rate: {r['overall_rate']:.1%}")
        print(f"  Fleiss kappa: {r['fleiss_kappa']:.3f} ({verdict_kappa(r['fleiss_kappa'])})")
        print(f"  Krippendorff alpha: {r['krippendorff_alpha']:.3f}")
        print(f"  Unanimous items: {r['unanimous_items']}/{r['n_items']} "
              f"({100*r['unanimous_items']/r['n_items']:.1f}%)")
        print(f"  per-model rate: "
              f"{', '.join(f'{MODEL_SHORT[m]}={r['per_model'].get(m, 0):.2f}' for m in MODELS)}")
        print(f"  per-eval rate: "
              f"{', '.join(f'{e[:10]}={r['per_evaluator'].get(e, 0):.2f}' for e in r['evaluators'])}")

    print(f"\nCombined: items={combined['n_items']}, "
          f"Fleiss={combined['fleiss_kappa']:.3f}, "
          f"unanimous={combined['unanimous_items']} "
          f"({100*combined['unanimous_items']/combined['n_items']:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binarisation",
        choices=sorted(BINARISATIONS.keys()),
        default="minor_or_major",
        help="How to collapse the 3-way flag into binary.  minor_or_major "
             "merges Minor+Major (default); major_only treats only Major as "
             "positive.",
    )
    args = parser.parse_args()

    rule = BINARISATIONS[args.binarisation]["rule"]
    out_dir = os.path.join(SCRIPT_DIR, "analysis_outputs", "hallucination", args.binarisation)
    os.makedirs(out_dir, exist_ok=True)

    per_round = []
    for slug, f in ROUND_FILES.items():
        df = load_and_binarize(f, rule)
        per_round.append(analyze_round(slug, df))
    combined = build_combined(per_round)

    save_pair_kappa_heatmap(per_round, os.path.join(out_dir, "01_pair_cohen_kappa.png"))
    save_vote_histogram(per_round, combined, os.path.join(out_dir, "02_vote_count_histogram.png"))
    save_rate_by_model_and_evaluator(per_round, os.path.join(out_dir, "03_rates_by_model_and_evaluator.png"))
    print(f"Saved figures to {out_dir}/")

    report_path = os.path.join(
        SCRIPT_DIR,
        f"HALLUCINATION_AGREEMENT_REPORT_{args.binarisation}.md",
    )
    write_report(per_round, combined, report_path, args.binarisation)
    print_console_summary(per_round, combined)


if __name__ == "__main__":
    main()
