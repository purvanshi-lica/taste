# TASTE analysis

The analysis framework behind the dataset paper. It characterizes the agreement
signal in the TASTE rankings and benchmarks off-the-shelf judges against the
designer panel. Three parts:

- `distribution_tests/` — the signal-validation framework. Per-prompt statistics
  (mean pairwise Kendall tau `T`, per-pair tau, majority-vote probability
  `p_max`, Condorcet cycle indicator) tested against exact and Monte-Carlo nulls
  at the fixed-panel design (p=4 models, R=5 raters), plus cross-domain reference
  anchors, sub-dimension signal ordering, and per-rater agreeableness.
- `vlm_judge/` — the open-weight VLM-as-judge benchmark: prompt construction,
  runners for several model families, and aggregation against the panel.
- `hallucination_agreement.py` — inter-rater agreement (Fleiss / Krippendorff)
  on the per-image hallucination flags.

## Install

```bash
pip install -r requirements.txt
```

`vlm_judge/` additionally needs the model stack from the repo root
(`torch`, `transformers`, `qwen-vl-utils`) and a CUDA GPU.

## Data

The masked ranking CSVs are fetched into the repo `data/` directory (see
`../data/README.md`). Designer identities are pre-masked to `A1`-`A5`
(Aesthetics) and `D1`-`D5` (Descriptions). The scripts read `data/` by default;
override with the `TASTE_DATA_DIR` environment variable.

## Reproducing the distribution tests

```bash
cd distribution_tests
python run_dimension.py                 # per-dimension stats -> stats/<slug>/stats.npz
python aggregate_report.py              # headline per-dimension report
python evaluator_agreeableness.py       # per-rater agreeableness
python pair_matrices.py                 # per-rater-pair structure
python verify.py                        # 10 internal consistency checks
python paper_figures/make_paper_figures.py        --out-dir figures
python paper_figures/make_anon_eval_figures.py    --out-dir figures
```

Figures and `stats/` are reproduction outputs and are git-ignored; rerunning
regenerates them. Headline numbers to check against are in
[`expected_results.md`](expected_results.md). Each script takes `--help`.

## Reproducing the VLM-judge benchmark

```bash
cd vlm_judge
python prepare_pairs.py                 # build the judge pair manifest from rankings
python run_vlm_judge.py --help          # run a model family (needs GPU + weights)
python aggregate_slate.py               # aggregate runs -> reports/slate_summary.*
python analyze_results.py               # per-criterion / position-bias analysis
```

Per-run `*.jsonl` outputs are git-ignored; the aggregated `reports/*.md`
summaries (model-level, no rater identities) are kept as documentation. The
machine-readable `reports/*.json` result dumps are not committed.

## Hallucination agreement

```bash
python hallucination_agreement.py                              # minor-or-major
python hallucination_agreement.py --binarisation major_only    # major only
```

## Cross-domain reference anchors (fetched separately)

`distribution_tests/refs/*.npy` are precomputed statistic arrays for the Sushi,
MovieLens, MT-Bench, and HPSv2-test reference distributions, each subsampled to
the (p=4, R=5) fixed-panel design (bootstrap B=50; for HPSv2-test the four
strongest text-to-image generators are held fixed and the COCO real image is
excluded, so the anchor reflects generated-only preference). The loaders that
built them depend on those third-party datasets and on an internal helper
package, neither of which is included, so the arrays are **fetched separately**
(alongside the dataset; see [`distribution_tests/refs/README.md`](distribution_tests/refs/README.md))
and are not committed to this repo. `generate_reference_stats.py` will only
regenerate them in an environment that has those loaders; the TASTE pipeline
above reproduces independently of them.
The anchors are descriptive visual references on the cross-domain figure, not
null hypotheses; the only formal tests are against the iid-uniform null.
