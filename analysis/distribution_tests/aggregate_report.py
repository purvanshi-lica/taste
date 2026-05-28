"""
Aggregate per-dimension distribution-test results into a single Markdown
report.  Formal tests are versus the iid-uniform null only; the three
reference datasets are descriptive anchors that appear in the per-dim
figures, not null hypotheses.

Usage:
    python aggregate_report.py \
        [--out ../DISTRIBUTION_TEST_REPORT.md]
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA_ANALYSIS_ROOT = HERE.parent
FIGURES_DIR = HERE / "figures"
STATS_DIR = HERE / "stats"

DIMENSIONS = [
    ("aesthetics_preference",    "UI+Ad Preference (holistic)", "Aesthetics"),
    ("aesthetics_mood",          "Mood & Tone Match",           "Aesthetics"),
    ("aesthetics_visual_hier",   "Visual Hierarchy",            "Aesthetics"),
    ("aesthetics_color_harmony", "Color Harmony",               "Aesthetics"),
    ("aesthetics_typography",    "Typography",                  "Aesthetics"),
    ("descriptions_preference",  "Preference (holistic)",       "Descriptions"),
    ("descriptions_color_acc",   "Color Accuracy",              "Descriptions"),
    ("descriptions_spatial_acc", "Spatial Accuracy",            "Descriptions"),
    ("descriptions_typography",  "Typography",                  "Descriptions"),
]
REFS = ["Sushi", "MovieLens", "HPSv2-test"]


def _p_fmt(p):
    if p is None or np.isnan(p):
        return "—"
    if p == 0:
        return "<1e-300"
    if p < 1e-3:
        return f"{p:.1e}"
    return f"{p:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",
                        default=str(DATA_ANALYSIS_ROOT / "DISTRIBUTION_TEST_REPORT.md"))
    args = parser.parse_args()

    summaries = {}
    tests_by_dim = {}
    for slug, display, group in DIMENSIONS:
        sumpath = STATS_DIR / slug / "summary.json"
        testpath = FIGURES_DIR / slug / "tests.json"
        if not sumpath.exists() or not testpath.exists():
            print(f"  [warn] missing results for {slug}")
            continue
        with open(sumpath) as f:
            summaries[slug] = json.load(f)
        with open(testpath) as f:
            tests_by_dim[slug] = json.load(f)

    if not summaries:
        raise SystemExit("No per-dimension results found. Run run_dimension.py first.")

    any_tests = next(iter(tests_by_dim.values()))
    cyc_null = any_tests["_cycle_null_rate"]
    ref_descr = any_tests["_reference_descriptive"]

    lines = []
    lines.append("# TASTE ranking data — distribution tests")
    lines.append("")
    lines.append("**Statistic framework:** fixed-panel (p=4, R=5) Kendall-tau battery "
                 "(exact null PMF of $T$ at $p=4, R=5$; Mahonian null for per-pair "
                 "Kendall $\\tau$; binomial $\\mathrm{Bin}(5, 0.5)$ null for majority-vote "
                 "probability; Monte-Carlo null rate for Condorcet cycles).")
    lines.append("")
    lines.append("**For every TASTE dimension** (80 prompts, 5 evaluators, 4 models):")
    lines.append("")
    lines.append("- $T$ per prompt = average of $C(5,2)=10$ pairwise Kendall $\\tau$ values")
    lines.append("- per-pair $\\tau$: 80 × 10 = 800 values on the Mahonian support "
                 "$\\{-1,-2/3,-1/3,0,1/3,2/3,1\\}$")
    lines.append("- majority-vote prob $p_{\\max}$: 80 × 6 = 480 values in "
                 "$\\{3/5, 4/5, 5/5\\}$")
    lines.append("- Condorcet cycle indicator: 80 binary values")
    lines.append("")
    lines.append(f"**Cycle-rate MC null (p=4, R=5):** {cyc_null:.4f}")
    lines.append("")
    lines.append("### Glossary")
    lines.append("")
    lines.append("- **$T$** — average pairwise Kendall $\\tau$ across the "
                 "$C(R,2)=10$ ranking-pairs in a single (prompt, evaluators) "
                 "sample.  Range $[-1/5, 1]$ for $p=4, R=5$.")
    lines.append("- **per-pair $\\tau$** — one Kendall $\\tau$ per ordered "
                 "ranking-pair (e.g. evaluator A vs evaluator B on prompt $k$); "
                 "7 discrete values for $p=4$.")
    lines.append("- **$p_{\\max}$** — majority-vote probability per item-pair: "
                 "$\\max(k/R, 1-k/R)$ where $k$ is the count of evaluators "
                 "putting item $a$ above item $b$.")
    lines.append("- **GOF** — *Goodness-of-Fit* (chi-squared test): "
                 "$\\chi^2 = \\sum_i (O_i - E_i)^2 / E_i$ where $O_i$ is the "
                 "observed bin count and $E_i = n \\cdot p_{i,\\text{null}}$. "
                 "Tests whether an observed histogram is compatible with a "
                 "hypothesised PMF.  A small $p$ rejects the hypothesis.")
    lines.append("- **Condorcet cycle** — when majority pairwise preferences "
                 "form an intransitive loop ($A \\succ B, B \\succ C, C \\succ A$). "
                 "Excess cycles signal cluster-style disagreement; sub-null "
                 "cycle rates signal weak common ordering.")
    lines.append("- **consensus alignment $\\bar\\tau_{\\text{others}}$** — "
                 "for one evaluator on one prompt, the mean Kendall $\\tau$ "
                 "between that evaluator and each of the other $R-1$ "
                 "evaluators.  Used in `disagreement.md` to flag outlier "
                 "evaluators on specific prompts.")
    lines.append("")
    lines.append("**Sushi, MovieLens, and HPSv2-test are visual anchors only.**  "
                 "They appear on every per-dimension figure so you can see what "
                 "real preference data at this design actually looks like.  "
                 "Sushi and MovieLens are non-image (food and movies); "
                 "HPSv2-test is general-domain image preference subsampled from "
                 "the HPDv2 test split (400 prompts × 10 fixed annotators × 9 "
                 "generative-model images per prompt; the COCO real-image at "
                 "index 9 is excluded so the anchor reflects generated-only "
                 "preference).  Of the 9 generative models we use only the "
                 "**top-4 by mean rank** — indices [1, 2, 6, 8] (mean ranks "
                 "1.59-2.59) which form a clean cluster ahead of the 5th-best "
                 "(mean rank 5.04).  This avoids the artifact where random "
                 "4-of-9 sampling pulls a much weaker generator (e.g. SD-v1.4) "
                 "into the comparison and inflates pairwise agreement.  These "
                 "4 models are held fixed across all 400 prompts; bootstrap "
                 "$B=50$ iterations vary only the 5-of-10 rater subset.  All "
                 "anchors save bootstrap-form arrays under `refs/`; only "
                 "HPSv2-test draws a 5-95% band on the figures (gated by "
                 "`REF_SHOW_BAND` — Sushi/MovieLens show only the mean line "
                 "to keep the plot readable).  None are null hypotheses — the "
                 "only formal tests here are versus the iid-uniform null.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # TL;DR
    lines.append("## TL;DR")
    lines.append("")
    best_dim = max(summaries.values(), key=lambda s: s["T_median"])
    worst_dim = min(summaries.values(), key=lambda s: s["T_median"])
    lines.append("Every TASTE dimension rejects the iid-uniform null on the "
                 "$T$, per-pair $\\tau$, and $p_{\\max}$ histograms (chi-squared GOF "
                 "$p < 10^{-10}$ everywhere), so there is detectable signal in every "
                 "dimension.")
    lines.append("")
    lines.append(f"- Strongest dimension: **{best_dim['display']} "
                 f"({best_dim['group']})** — median $T={best_dim['T_median']:+.3f}$")
    lines.append(f"- Weakest dimension: **{worst_dim['display']} "
                 f"({worst_dim['group']})** — median $T={worst_dim['T_median']:+.3f}$")
    lines.append("")
    lines.append("For orientation, the reference medians (all $p=4, R=5$):")
    lines.append("")
    lines.append("| Reference (descriptive only) | $n$ samples | $T$ median |")
    lines.append("|---|---|---|")
    for name in REFS:
        d = ref_descr[name]
        lines.append(f"| {name} | {d['n']:,} | {d['T_median']:+.3f} |")
    lines.append(f"| TASTE range across 9 dims | — | "
                 f"{worst_dim['T_median']:+.3f} to {best_dim['T_median']:+.3f} |")
    lines.append("")
    lines.append(f"All TASTE dimensions sit between the Sushi median "
                 f"({ref_descr['Sushi']['T_median']:+.3f}) and the MovieLens median "
                 f"({ref_descr['MovieLens']['T_median']:+.3f}) — i.e. roughly "
                 f"the same statistical character as ranking foods or movies "
                 f"(both subjective-with-mild-objectivity tasks).  "
                 f"HPSv2-test on the top-4 generative models "
                 f"(median $T={ref_descr['HPSv2-test']['T_median']:+.3f}$) sits "
                 f"a step above MovieLens — modestly higher agreement on "
                 f"comparing the four strongest text-to-image generators "
                 f"head-to-head than on subjective taste tasks.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- Per-dimension headline ----
    lines.append("## 1. Headline table — per TASTE dimension")
    lines.append("")
    lines.append("| Dimension | Group | $n$ | $T$ median | per-pair $\\tau$ mean "
                 "| $p_{\\max}$ mean | cycle rate |")
    lines.append("|---|---|---|---|---|---|---|")
    for slug, display, group in DIMENSIONS:
        if slug not in summaries:
            continue
        s = summaries[slug]
        lines.append(f"| {display} | {group} | {s['n_prompts']} | "
                     f"{s['T_median']:+.3f} | {s['pair_tau_mean']:+.3f} | "
                     f"{s['pmaj_mean']:.3f} | {s['cycle_rate']:.3f} |")
    lines.append("")
    lines.append("**Reference distributions** (descriptive stats only, shown on the "
                 "per-dim plots as visual anchors — NOT null hypotheses):")
    lines.append("")
    lines.append("| Reference | $n$ | $T$ median | per-pair $\\tau$ mean "
                 "| $p_{\\max}$ mean | cycle rate |")
    lines.append("|---|---|---|---|---|---|")
    for name in REFS:
        d = ref_descr[name]
        lines.append(f"| {name} | {d['n']:,} | {d['T_median']:+.3f} | "
                     f"{d['pair_tau_mean']:+.3f} | {d['pmaj_mean']:.3f} | "
                     f"{d['cycle_rate']:.3f} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- GOF vs null ----
    lines.append("## 2. Chi-squared GOF vs the iid-uniform null")
    lines.append("")
    lines.append("The null is \"the 5 evaluators are ranking items at random and the "
                 "agreement we see is sampling noise.\"  A small $p$ rejects this.")
    lines.append("")
    lines.append("| Dimension | $T$ | per-pair $\\tau$ | $p_{\\max}$ | cycle (binomial) |")
    lines.append("|---|---|---|---|---|")
    for slug, display, group in DIMENSIONS:
        if slug not in tests_by_dim:
            continue
        t = tests_by_dim[slug]
        lines.append(f"| {display} | {_p_fmt(t['T']['null']['p'])} | "
                     f"{_p_fmt(t['pair_tau']['null']['p'])} | "
                     f"{_p_fmt(t['pmaj']['null']['p'])} | "
                     f"{_p_fmt(t['cycle']['null']['p'])} |")
    lines.append("")
    lines.append("*$T$, per-pair $\\tau$, and $p_{\\max}$ reject the null everywhere. "
                 "The cycle-rate binomial test at $n=80$ does not reject the null for "
                 "most dimensions — observed cycle rates sit near or below the null "
                 "rate (0.21), consistent with a weak Mallows-type common-ordering "
                 "signal rather than cluster alternatives (which would produce excess "
                 "cycles).*")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- Per-pair reviewer view (cohort-level) ----
    lines.append("## 3. Reviewer-pair structure — agreement, grouping, outliers")
    lines.append("")
    lines.append("Cohort-level views pool each evaluator pair's per-prompt $\\tau$ "
                 "values across all dimensions in the cohort (5 dims × 80 prompts "
                 "= 400 values per pair for Aesthetics, 4 × 80 = 320 for Descriptions).")
    lines.append("")
    lines.append("- **5×5 mean-$\\tau$ heatmap** identifies clusters and outliers.")
    lines.append("- **10-pair distribution grid** shows the full shape of each pair's "
                 "agreement, sorted from most-agreeing (top-left) to least-agreeing "
                 "(bottom-right) and overlaid on the Mahonian null.")
    lines.append("")
    lines.append("Aesthetics cohort:")
    lines.append("")
    lines.append("- Heatmap: `distribution_tests/figures/_pair_heatmap_aesthetics.png`")
    lines.append("- Distribution grid: `distribution_tests/figures/_pair_distribution_aesthetics.png`")
    lines.append("")
    lines.append("Descriptions cohort:")
    lines.append("")
    lines.append("- Heatmap: `distribution_tests/figures/_pair_heatmap_descriptions.png`")
    lines.append("- Distribution grid: `distribution_tests/figures/_pair_distribution_descriptions.png`")
    lines.append("")
    lines.append("**Reading the figures:**")
    lines.append("")
    lines.append("- *Aesthetics:* the four higher-agreement raters (A2–A5) form a tight "
                 "cluster (pairwise τ ≈ 0.15–0.20).  A1 is an outlier — "
                 "its τ against every other evaluator is in the 0.06–0.09 range.  "
                 "Look at the distribution grid: the top 5 pairs (excluding A1) "
                 "show a clear right-shift on the Mahonian null; the bottom 5 "
                 "pairs (all containing A1) sit near the null.")
    lines.append("- *Descriptions:* softer pattern.  the top rater-pair agrees most "
                 "(τ = +0.259), the second pair at +0.240.  the weakest pair "
                 "is +0.105 — D1 is the soft outlier here, "
                 "with low τ against several peers.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- Per-evaluator (5 numbers per cohort) ----
    lines.append("## 4. Per-evaluator agreeableness (cohort-level summary)")
    lines.append("")
    lines.append("Single number per (cohort, evaluator): mean pairwise $\\tau$ "
                 "averaged over the 4 partner-pairs the evaluator participates "
                 "in × all 80 prompts × all dimensions in the cohort.  High = "
                 "consistently agrees with peers; low = idiosyncratic preferences.")
    lines.append("")
    agr_path = HERE / "figures" / "_evaluator_agreeableness.md"
    if agr_path.exists():
        with open(agr_path) as f:
            agr_text = f.read()
        # Strip the top "# Per-evaluator agreeableness" h1 since we're
        # nesting; keep the cohort sections.
        agr_text = "\n".join(line for line in agr_text.splitlines()
                              if not line.startswith("# Per-evaluator"))
        lines.append(agr_text.strip())
        lines.append("")
        lines.append("Mean heatmap + bar-chart: "
                     "`distribution_tests/figures/_evaluator_agreeableness_*.png`. "
                     "Per-evaluator distribution grid (richer view: each cell is "
                     "a histogram of 320 tau values, not just a mean): "
                     "`distribution_tests/figures/_evaluator_distribution_*.png`.")
        lines.append("")
    else:
        lines.append("*(Run `python evaluator_agreeableness.py` to generate.)*")
        lines.append("")
    lines.append("---")
    lines.append("")

    # ---- Per-dim output links ----
    lines.append("## 5. Per-dimension outputs")
    lines.append("")
    lines.append("Each dimension has its own directory under "
                 "`distribution_tests/figures/<slug>/`:")
    lines.append("")
    lines.append("- `main.png` — 4-panel figure (T, pair-$\\tau$, $p_{\\max}$, "
                 "cycle-rate) with null in grey and Sushi / MovieLens "
                 "overlaid as lines for reference.")
    lines.append("- `pair_reviewers.png` — 2x5 grid of pairwise $\\tau$ histograms "
                 "by evaluator pair (10 pairs, sorted by mean $\\tau$).")
    lines.append("- `tests.json` — GOF-vs-null p-values + reference descriptive stats.")
    lines.append("- `disagreement.md` — top-5 highest-disagreement prompts (lowest $T$) "
                 "with inline Cloudinary thumbnails + per-evaluator rank matrix.")
    lines.append("")
    for slug, display, group in DIMENSIONS:
        if slug not in summaries:
            continue
        lines.append(f"- **{display}** ({group}) — "
                     f"`distribution_tests/figures/{slug}/` "
                     f"· [main](distribution_tests/figures/{slug}/main.png) "
                     f"· [pair grid](distribution_tests/figures/{slug}/pair_reviewers.png) "
                     f"· [disagreement](distribution_tests/figures/{slug}/disagreement.md)")
    lines.append("")

    # ---- Interpretation notes ----
    lines.append("---")
    lines.append("")
    lines.append("## 6. Interpretation notes")
    lines.append("")
    lines.append("**Why histogram GOF rather than a single \"is $\\tau$ big enough?\" "
                 "threshold.**  A single summary like \"mean $\\tau = 0.16$\" hides the "
                 "shape of agreement.  Two datasets with the same mean can have very "
                 "different per-prompt profiles (many near-unanimous + many near-reversed "
                 "vs. a broad middle) which matters for downstream modelling.  The exact "
                 "null PMF at $(p=4, R=5)$ gives a principled comparator, and the chi-sq "
                 "GOF formalises \"is the full histogram shape compatible with "
                 "sampling-noise agreement?\"")
    lines.append("")
    lines.append("**Why $p_{\\max}$ rather than per-prompt entropy.**  Entropy $H(p)$ is "
                 "a one-to-one function of $p = k/R$ given $R$, so the information content "
                 "is identical.  But $p_{\\max} = \\max(k/R, 1-k/R)$ has three discrete "
                 "values for $R=5$ and an explicit binomial null "
                 "($P(3/5)=20/32, P(4/5)=10/32, P(5/5)=2/32$), making histogram-vs-null "
                 "a direct test.")
    lines.append("")
    lines.append("**Why the cycle rate should also be tested.**  Cycles are rare under "
                 "5 iid uniform rankings of 4 items (~21% per-prompt MC).  Excess cycles "
                 "would suggest a mixture-of-orderings (factions); sub-random cycles "
                 "would suggest a weak common ordering.  The binomial test at $n=80$ "
                 "is the direct inference.  *(The legacy `sub_dimension_analysis.py` "
                 "reported 75%+ cycle rates because a bug in its `prefs[(y,x)]` "
                 "assignment inverted winner and loser — it was counting transitive "
                 "chains, not cycles.  The corrected rates are all below 20%.)*")
    lines.append("")
    lines.append("**Per-evaluator-pair matrices (`pair_reviewers.png`)** are the most "
                 "actionable view: sort C(5,2)=10 pairs by mean $\\tau$ descending to "
                 "see which evaluators agree most consistently and which are outliers. "
                 "In Aesthetics UI+Ad Preference, the top rater-pair tops at mean $\\tau = "
                 "+0.229$; the bottom pair (containing outlier A1) is at $+0.067$.")
    lines.append("")
    lines.append("**Disagreement exemplars (`disagreement.md`)** surface the 5 lowest-$T$ "
                 "prompts per dimension with inline 280-pixel Cloudinary thumbnails of "
                 "the 4 model outputs and the evaluator-by-model rank matrix.  Click any "
                 "thumbnail for the full-resolution original.  These are the prompts to "
                 "eyeball manually to understand *why* designers disagree — subjective "
                 "criteria, hallucinations, genre-specific preferences, etc.")
    lines.append("")

    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
