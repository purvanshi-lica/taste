"""
Build per-dimension 5x5 reviewer-vs-reviewer matrices and a unified
markdown report identifying most-agreeing and most-disagreeing reviewers
per dimension.

For each of the 9 TASTE dimensions (5 evaluators, 80 prompts):
  - 5x5 matrix of mean pairwise Kendall tau (with std in brackets) per
    evaluator pair
  - 5x5 matrix of chi-squared GOF p-values: per pair, snap the 80
    per-prompt tau values onto the 7-point Mahonian support and run
    chi2 GOF vs the null PMF.  Small p = pair's distribution is *not*
    consistent with iid-uniform noise (i.e. the pair has structured
    agreement signal); large p = pair looks like random.
  - Identify the most-agreeing pair (highest mean tau) and the
    most-disagreeing pair (lowest mean tau) per dimension.

Diagonal entries are not meaningful (an evaluator vs itself) and are
rendered as "--".

Output:
  figures/_pair_matrices.md            top-level summary + per-dim sections
  figures/<slug>/pair_matrices.md      single-dimension matrices

Reads from the per-dimension caches under stats/<slug>/stats.npz
(produced by taste_stats.py).
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from taste_stats import DIMENSIONS  # noqa: E402
from distribution_plots import (  # noqa: E402
    chi2_gof_vs_null, snap_to_grid, pairtau_null_grid,
)


SHORT = lambda name: name.split()[0][:10]  # noqa: E731


def load_dimension_npz(slug):
    path = HERE / "stats" / slug / "stats.npz"
    return np.load(path, allow_pickle=False)


def per_pair_table(npz):
    """Recover per-pair (evaluator_i, evaluator_j) mappings + tau columns.

    Returns:
        evaluators: ordered list of the 5 evaluator names in the cohort
        pair_idx_map: {(eval_i_idx, eval_j_idx): col_in_pair_tau} for
            each unordered pair (i < j)
        pair_tau: (n_prompts, 10) array
    """
    evaluators = list(npz["evaluators"])
    pair_a = list(npz["pair_labels_a"])
    pair_b = list(npz["pair_labels_b"])
    pair_tau = npz["pair_tau"]                     # (n_prompts, 10)
    n_pairs = pair_tau.shape[1]
    pair_idx_map = {}
    for col in range(n_pairs):
        i = evaluators.index(pair_a[col])
        j = evaluators.index(pair_b[col])
        if i > j:
            i, j = j, i
        pair_idx_map[(i, j)] = col
    return evaluators, pair_idx_map, pair_tau


def build_matrices(npz):
    """Compute the 5x5 tau-summary and GOF-p matrices for one dim.

    Returns dict with:
        evaluators
        mean_mat / median_mat / std_mat / min_mat / max_mat   (5,5) NaN on diagonal
        gof_p_mat / gof_chi2_mat / gof_df_mat                 (5,5) NaN on diagonal
        per_pair_summary: list of dicts ordered by mean tau descending
    """
    evaluators, pair_idx_map, pair_tau = per_pair_table(npz)
    n = len(evaluators)
    mean_mat = np.full((n, n), np.nan)
    median_mat = np.full((n, n), np.nan)
    std_mat = np.full((n, n), np.nan)
    min_mat = np.full((n, n), np.nan)
    max_mat = np.full((n, n), np.nan)
    gof_p_mat = np.full((n, n), np.nan)
    gof_chi2_mat = np.full((n, n), np.nan)
    gof_df_mat = np.full((n, n), np.nan)

    tau_grid, tau_null = pairtau_null_grid(4)
    per_pair_rows = []
    for (i, j), col in sorted(pair_idx_map.items()):
        vals = pair_tau[:, col]
        m = float(vals.mean())
        med = float(np.median(vals))
        s = float(vals.std(ddof=1))
        lo = float(vals.min())
        hi = float(vals.max())
        counts = snap_to_grid(vals, tau_grid)
        chi2, pv, df, _nb = chi2_gof_vs_null(counts, tau_null)
        mean_mat[i, j] = mean_mat[j, i] = m
        median_mat[i, j] = median_mat[j, i] = med
        std_mat[i, j] = std_mat[j, i] = s
        min_mat[i, j] = min_mat[j, i] = lo
        max_mat[i, j] = max_mat[j, i] = hi
        gof_p_mat[i, j] = gof_p_mat[j, i] = pv
        gof_chi2_mat[i, j] = gof_chi2_mat[j, i] = chi2
        gof_df_mat[i, j] = gof_df_mat[j, i] = df
        per_pair_rows.append({
            "i": i, "j": j,
            "ev_i": evaluators[i], "ev_j": evaluators[j],
            "mean_tau": m, "median_tau": med, "std_tau": s,
            "min_tau": lo, "max_tau": hi,
            "n": int(len(vals)),
            "gof_chi2": chi2, "gof_p": pv, "gof_df": int(df),
        })
    per_pair_rows.sort(key=lambda r: -r["mean_tau"])
    return {
        "evaluators": evaluators,
        "mean_mat": mean_mat, "median_mat": median_mat, "std_mat": std_mat,
        "min_mat": min_mat, "max_mat": max_mat,
        "gof_p_mat": gof_p_mat,
        "gof_chi2_mat": gof_chi2_mat,
        "gof_df_mat": gof_df_mat,
        "per_pair_summary": per_pair_rows,
    }


# ---------- markdown rendering ----------

def _fmt_tau_summary_cell(m, med, s, lo, hi):
    """Tau cell with five stats: mean, median, std, min, max.
    Multi-line cell using <br> (renders correctly in GitHub-flavored markdown).
    """
    if np.isnan(m):
        return "--"
    line1 = f"mean={m:+.3f}, med={med:+.3f}"
    line2 = f"sd={s:.3f}, [{lo:+.2f}, {hi:+.2f}]"
    return f"{line1}<br>{line2}"


def _fmt_gof_cell(chi2, pv, df):
    """Distribution-test cell: 'chi2=X.X, p=Y' on a single line."""
    if np.isnan(chi2):
        return "--"
    if pv < 1e-300:
        pstr = "<1e-300"
    elif pv < 0.001:
        pstr = f"{pv:.1e}"
    else:
        pstr = f"{pv:.3f}"
    return f"$\\chi^2$={chi2:.1f}, p={pstr}"


def render_matrix_table(matrix_fn, evaluators, header_label):
    """Render a square matrix as a markdown table with cells from matrix_fn(i,j)."""
    short = [SHORT(e) for e in evaluators]
    lines = ["| " + header_label + " | " + " | ".join(short) + " |"]
    lines.append("|---" + "|---" * len(evaluators) + "|")
    for i, e in enumerate(evaluators):
        row = [short[i]]
        for j in range(len(evaluators)):
            row.append(matrix_fn(i, j))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_dim_section(slug, display, group, M):
    """Per-dimension section: title + 2 matrices.  Nothing else."""
    evaluators = M["evaluators"]
    lines = []
    lines.append(f"#### {display}")
    lines.append("")

    # Matrix A: tau summary (mean, median, std, min, max)
    lines.append("**Pairwise Kendall $\\tau$ — mean / median / std / [min, max] "
                 "across 80 per-prompt $\\tau$ values:**")
    lines.append("")
    lines.append(render_matrix_table(
        lambda i, j: _fmt_tau_summary_cell(
            M["mean_mat"][i, j], M["median_mat"][i, j],
            M["std_mat"][i, j], M["min_mat"][i, j], M["max_mat"][i, j],
        ),
        evaluators, header_label=r"$\tau$ stats",
    ))
    lines.append("")

    # Matrix B: distribution-test (chi-squared GOF vs Mahonian null)
    lines.append("**Distribution test — chi-squared GOF vs Mahonian null "
                 "(per-pair $\\tau$-histogram on the 7-point support, "
                 "df=6):**")
    lines.append("")
    lines.append(render_matrix_table(
        lambda i, j: _fmt_gof_cell(M["gof_chi2_mat"][i, j],
                                    M["gof_p_mat"][i, j],
                                    M["gof_df_mat"][i, j]),
        evaluators, header_label=r"$\chi^2$, $p$",
    ))
    lines.append("")
    return "\n".join(lines)


def render_top_summary(all_dims):
    """Headline table across all 9 dimensions.

    For each dim, the top and bottom pair, plus min and max GOF p-value.
    """
    lines = []
    lines.append("### Most-agreeing / most-disagreeing pair per dimension")
    lines.append("")
    lines.append("| Dimension | Group | Most-agreeing pair | $\\bar\\tau$ "
                 "(sd) | Most-disagreeing pair | $\\bar\\tau$ (sd) | "
                 "$\\Delta\\bar\\tau$ |")
    lines.append("|---|---|---|---|---|---|---|")
    for slug, display, group, M in all_dims:
        pp = M["per_pair_summary"]
        top, bot = pp[0], pp[-1]
        spread = top["mean_tau"] - bot["mean_tau"]
        lines.append(
            f"| {display} | {group} | "
            f"{SHORT(top['ev_i'])} × {SHORT(top['ev_j'])} | "
            f"{top['mean_tau']:+.3f} ({top['std_tau']:.3f}) | "
            f"{SHORT(bot['ev_i'])} × {SHORT(bot['ev_j'])} | "
            f"{bot['mean_tau']:+.3f} ({bot['std_tau']:.3f}) | "
            f"{spread:+.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_pair_freq_summary(all_dims):
    """No-op kept for backwards-compat.  We removed this section because
    the headline table already shows top/bottom pair per dim."""
    return ""


def render_evaluator_outlier_table(all_dims):
    """Per-evaluator participation in the bottom pair (most-disagreeing).

    Per cohort, count how often each of the 5 evaluators shows up in the
    bottom pair.  An evaluator appearing 4+ times out of N dimensions is
    a consistent outlier."""
    lines = []
    lines.append("### Per-evaluator outlier rate by cohort")
    lines.append("")
    lines.append("How often each evaluator appears in the **most-agreeing** "
                 "and **most-disagreeing** pair across their cohort's "
                 "dimensions.  An evaluator that's consistently in the "
                 "bottom pair has systematically different preferences "
                 "from peers.")
    lines.append("")

    by_cohort = {}
    for slug, display, group, M in all_dims:
        by_cohort.setdefault(group, []).append((display, M))

    for group, dims in by_cohort.items():
        lines.append(f"**{group} cohort  ({len(dims)} dimensions)**")
        lines.append("")
        evaluators = dims[0][1]["evaluators"]
        ev_count_bot = {e: 0 for e in evaluators}
        ev_count_top = {e: 0 for e in evaluators}
        for display, M in dims:
            pp = M["per_pair_summary"]
            top, bot = pp[0], pp[-1]
            ev_count_top[top["ev_i"]] += 1
            ev_count_top[top["ev_j"]] += 1
            ev_count_bot[bot["ev_i"]] += 1
            ev_count_bot[bot["ev_j"]] += 1
        lines.append("| Evaluator | "
                     "appearances in **most-agreeing** pair | "
                     "appearances in **most-disagreeing** pair |")
        lines.append("|---|---|---|")
        # Sort by descending bottom-count (most disagreeing first)
        ordered = sorted(evaluators, key=lambda e: -ev_count_bot[e])
        for e in ordered:
            lines.append(f"| {e} | {ev_count_top[e]}/{len(dims)} | "
                         f"{ev_count_bot[e]}/{len(dims)} |")
        lines.append("")
    return "\n".join(lines)


def main():
    out_root = HERE / "figures"
    out_root.mkdir(exist_ok=True)

    # Process all 9 dimensions
    all_dims = []
    for slug, display, group, csv_name in DIMENSIONS:
        npz = load_dimension_npz(slug)
        M = build_matrices(npz)
        all_dims.append((slug, display, group, M))
        # Also write a single-dimension fragment under figures/<slug>/
        per_dim_md = (out_root / slug / "pair_matrices.md")
        per_dim_md.parent.mkdir(parents=True, exist_ok=True)
        with open(per_dim_md, "w") as f:
            f.write(f"# Pair matrices — {display}  ({group})\n\n")
            f.write(render_dim_section(slug, display, group, M))
        print(f"  {slug:28s} wrote {per_dim_md}")

    # Master report
    parts = []
    parts.append("# Reviewer-pair agreement matrices (per dimension)")
    parts.append("")
    parts.append("Two 5×5 reviewer-vs-reviewer matrices per TASTE dimension: "
                 "(1) mean pairwise Kendall $\\tau$ ± std across the 80 "
                 "per-prompt values, and (2) chi-squared GOF $p$-value of "
                 "the pair's $\\tau$-histogram against the Mahonian null "
                 "at $(p=4,\\,R=2)$.  Small $p$ ⇒ the pair has structured "
                 "agreement; large $p$ ⇒ the pair's distribution looks "
                 "like random ranking.  Diagonal cells are `--` (an "
                 "evaluator vs. themselves is trivially $\\tau=1$).")
    parts.append("")
    parts.append("Aesthetics raters are labelled A1–A5 and Descriptions raters "
                 "D1–D5, assigned by mean pairwise Kendall tau ascending within "
                 "each cohort (index 1 = lowest-agreement).  Cohort pools are "
                 "disjoint, so A1 and D1 are different people.")
    parts.append("")

    # ---- Summary block at the top ----
    parts.append("## Summary")
    parts.append("")
    parts.append(render_top_summary(all_dims))
    parts.append(render_evaluator_outlier_table(all_dims))

    parts.append("---")
    parts.append("")
    parts.append("## Per-dimension matrices")
    parts.append("")

    # Order: aesthetics first, descriptions second
    by_cohort = {}
    for slug, display, group, M in all_dims:
        by_cohort.setdefault(group, []).append((slug, display, group, M))
    for group in ["Aesthetics", "Descriptions"]:
        if group not in by_cohort:
            continue
        parts.append(f"### {group} cohort")
        parts.append("")
        for slug, display, group_, M in by_cohort[group]:
            parts.append(render_dim_section(slug, display, group_, M))
            parts.append("")

    out_path = out_root / "_pair_matrices.md"
    with open(out_path, "w") as f:
        f.write("\n".join(parts))
    print(f"\nWrote master report: {out_path}")

    # Also dump per-pair summaries to JSON for downstream use
    json_payload = {}
    for slug, display, group, M in all_dims:
        json_payload[slug] = {
            "display": display, "group": group,
            "evaluators": M["evaluators"],
            "mean_mat":   M["mean_mat"].tolist(),
            "median_mat": M["median_mat"].tolist(),
            "std_mat":    M["std_mat"].tolist(),
            "min_mat":    M["min_mat"].tolist(),
            "max_mat":    M["max_mat"].tolist(),
            "gof_p_mat":  M["gof_p_mat"].tolist(),
            "gof_chi2_mat": M["gof_chi2_mat"].tolist(),
            "gof_df_mat": M["gof_df_mat"].tolist(),
            "per_pair_summary": M["per_pair_summary"],
        }
    with open(out_root / "_pair_matrices.json", "w") as f:
        json.dump(json_payload, f, indent=2, default=float)
    print(f"Wrote JSON payload: {out_root / '_pair_matrices.json'}")


if __name__ == "__main__":
    main()
