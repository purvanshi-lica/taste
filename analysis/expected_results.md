# Expected results

Headline numbers for verifying a reproduction run. Dimension-level statistics
only (no rater identities). Produced by `distribution_tests/run_dimension.py`
followed by `aggregate_report.py`; regenerated values should match these.

## Per-dimension distribution statistics (p=4, R=5; 80 prompts each)

| Dimension | Cohort | n | T median | per-pair tau mean | p_max mean | cycle rate |
|---|---|---|---|---|---|---|
| UI+Ad Preference (holistic) | Aesthetics | 80 | +0.133 | +0.159 | 0.744 | 0.150 |
| Mood & Tone Match | Aesthetics | 80 | +0.133 | +0.147 | 0.737 | 0.125 |
| Visual Hierarchy | Aesthetics | 80 | +0.133 | +0.128 | 0.734 | 0.087 |
| Color Harmony | Aesthetics | 80 | +0.067 | +0.103 | 0.723 | 0.138 |
| Typography | Aesthetics | 80 | +0.133 | +0.119 | 0.733 | 0.075 |
| Preference (holistic) | Descriptions | 80 | +0.133 | +0.163 | 0.745 | 0.150 |
| Color Accuracy | Descriptions | 80 | +0.133 | +0.144 | 0.741 | 0.113 |
| Spatial Accuracy | Descriptions | 80 | +0.200 | +0.182 | 0.750 | 0.150 |
| Typography | Descriptions | 80 | +0.200 | +0.224 | 0.767 | 0.062 |

- Every dimension rejects the iid-uniform null on the T, per-pair tau, and
  p_max histograms (chi-squared goodness-of-fit p < 1e-10 throughout).
- Strongest signal: Typography (Descriptions), median T = +0.200,
  per-pair tau mean +0.224. Weakest: Color Harmony (Aesthetics), median
  T = +0.067.
- Monte-Carlo cycle-rate null at (p=4, R=5): 0.2113. All dimension cycle rates
  sit below this null.

## Cross-domain reference anchors (fetched separately; `refs/*.npy`)

| Reference | n samples | T median | per-pair tau mean | p_max mean | cycle rate |
|---|---|---|---|---|---|
| Sushi | 20,000 | +0.133 | +0.144 | 0.739 | 0.107 |
| MovieLens | 24,000 | +0.200 | +0.216 | 0.764 | 0.146 |
| HPSv2-test | 20,000 | +0.267 | +0.302 | 0.790 | 0.060 |

TASTE dimensions span +0.067 to +0.200 in median T, sitting between the Sushi
and MovieLens anchors (subjective-with-mild-objectivity tasks) and below the
top-4-generative HPSv2-test anchor (+0.267).

## Hallucination-flag agreement

Inter-rater agreement on the binary hallucination flags is lower than on the
rankings: Fleiss / Krippendorff agreement is approximately 0.12 for the
major-only binarisation, against a ranking Krippendorff alpha of approximately
0.19. The "objective" binary task is noisier than the ranking task, not easier.
Regenerate with `hallucination_agreement.py` (both binarisations).
