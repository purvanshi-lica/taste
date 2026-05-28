# Distribution-based tests for TASTE ranking data

Histogram tests for per-prompt Kendall $T$, individual pairwise $\tau$,
majority-vote probability, and Condorcet-cycle rate.

**Formal tests are versus the iid-uniform null only** (exact null PMF at
$(p=4, R=5)$ fixed-panel design).

Sushi, MovieLens, and MT-Bench appear on every per-dimension plot as
**visual anchors** (so you can eyeball what real preference data at the
same design looks like) — they are NOT null hypotheses and no formal
two-sample tests are run against them.

A 10-check verification suite (`verify.py`) confirms the null PMF moments
and tail probabilities match the paper, the reference T quantiles match
main.tex Table 4, the cycle detector matches a textbook reference
implementation, and the chi-squared GOF is well-calibrated (see bottom of
this README for the full checklist).

## Files

| File | Role |
|---|---|
| `null_kendall.py` | Exact null machinery — for the (p=4, R=5) fixed-panel design. |
| `generate_reference_stats.py` | Re-derives per-sample $T$ / pairwise $\tau$ / $p_{\max}$ / cycle arrays for the three reference datasets (imports the external-dataset loaders lazily). Writes `refs/<ds>_*.npy`. |
| `taste_stats.py` | Loads one TASTE CSV into a `(n_prompts, 5, 4)` rank tensor and computes the four statistics per prompt. Writes `stats/<slug>/stats.npz`. |
| `distribution_plots.py` | Plotting helpers + the chi-squared / KS / binomial tests. |
| `run_dimension.py` | Per-dimension driver: main 4-panel figure + 2×5 per-pair-reviewer figure + `tests.json` + `disagreement.md`. |
| `aggregate_report.py` | Rolls all 9 dimensions into `../DISTRIBUTION_TEST_REPORT.md`. |

## Reproducing

```bash
# 1. one-time: build the reference arrays (takes ~30s)
python generate_reference_stats.py

# 2. one-time: TASTE per-dimension npz caches
python taste_stats.py

# 3. all 9 TASTE dimensions (each: main.png, pair_reviewers.png, tests.json,
#    disagreement.md under figures/<slug>/)
python run_dimension.py

# 4. master markdown report
python aggregate_report.py
# -> ../DISTRIBUTION_TEST_REPORT.md
```

## What each figure shows

### `figures/<slug>/main.png` — 4-panel "signal" figure

- **(a) per-prompt $T$ histogram** on the exact null support, overlaid with null PMF (grey bars), TASTE (black bars), and the three reference empiricals (colored lines). Where TASTE matches a reference, their lines track the black bars.
- **(b) individual pairwise $\tau$** on the 7-point Mahonian support $\{-1, -2/3, -1/3, 0, +1/3, +2/3, +1\}$. Same overlay logic.
- **(c) majority-vote probability** $p_{\max} = \max(k/5, 1-k/5) \in \{3/5, 4/5, 5/5\}$. Null PMF is $(20/32, 10/32, 2/32)$.
- **(d) Condorcet cycle rate** — binary per-prompt indicator, null MC'd to ~21.1%.

Each panel reports the chi-squared GOF $p$-value against the null.

### `figures/<slug>/pair_reviewers.png` — evaluator-pair tau grid

2×5 grid of histograms, one per $C(5,2)=10$ evaluator pair, sorted by mean $\tau$ descending. Each histogram has 80 observations (one per prompt) on the 7-point Mahonian support, overlaid on the Mahonian null (grey). Identifies which reviewer pairs agree more than the average and which pair up as outliers.

Aesthetic and Descriptions dimensions have disjoint evaluator pools, so each figure is self-contained within its cohort.

### `figures/<slug>/disagreement.md` — top-5 highest-disagreement prompts

Ranked by ascending per-prompt $T$. For each: Cloudinary-thumbnailed images of all 4 model outputs (280px, click for full-res), prompt preview, and the per-evaluator × per-model rank matrix.

### `figures/<slug>/tests.json` — machine-readable stats

All chi-squared, KS, and binomial $p$-values, plus raw $n$s and cycle null rate.

## Key design choices

- **$T$ vs individual $\tau$:** both are reported. $T$ is cleaner per prompt (one number). Individual $\tau$ exposes the underlying per-rater-pair concordance shape.
- **$p_{\max}$ instead of entropy:** entropy $H(p)$ is a one-to-one function of $p = k/R$, so same information content, but $p_{\max}$ is easier to reason about and has an explicit binomial null.
- **Cycle rate via binary indicator:** matches the "is this prompt coherent" question.  The old triple-rate definition was prone to a bug in the legacy code; this binary per-prompt version is simpler and statistically cleaner.
- **Null $p$-values are small everywhere** for the three histogram statistics — all 9 TASTE dimensions reject iid-uniform on $T$, pair-$\tau$, and $p_{\max}$.  The cycle-rate binomial test at $n=80$ is less powerful and mostly fails to reject, since TASTE cycle rates sit near or below the null rate (0.21).

## Verification (`verify.py`)

Run `python verify.py` to run all 10 checks:

| # | Check | Verifies |
|---|---|---|
| V1 | Null PMF moments & tails | Exact null mean=0, var=13/540, skew=1.041, excess kurt=1.049; upper-tails at T=1/3, 2/5, 2/3 match main.tex to 6 decimals. |
| V2 | Cycle detector on 4 constructed cases | all-identical, 3v2 opposing, constructed Condorcet cycle, 4v1 dissent. |
| V3 | Cycle null rate MC | Fresh independent seed MC gives the same rate (0.21) to within 0.003. |
| V4 | Reference T quantiles | Sushi/MovieLens/MT-Bench mean/median/P95 match main.tex Table 4. |
| V5 | TASTE T via scipy.stats.kendalltau | Per-prompt T from my `stat_T` matches scipy per-pair Kendall tau + average to 1e-16. |
| V6 | $p_{\max}$ consistency | matches raw-tensor recomputation exactly; support is exactly $\{3/5, 4/5, 5/5\}$. |
| V7 | Chi-sq GOF self-consistency | 5000 null MC samples → p-values are uniform (KS p > 0.001) and rejection rate at $\alpha=0.05$ is within 0.015 of 0.05. |
| V8 | Cycle vs textbook reference | All 80 aesthetics_preference prompts agree with a completely separate textbook implementation. |
| V9 | mean(pair_tau) == T | Exact identity (avg of C(R,2) per-pair taus equals the T statistic). |
| V10 | Null grid coverage | Every TASTE and reference T value snaps to an exact null-PMF support point (within 1e-16 rounding). |

All 10 pass.
