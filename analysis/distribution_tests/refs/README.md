# distribution_tests/refs/

Cross-domain reference-anchor arrays (`*.npy`) for the Sushi, MovieLens,
MT-Bench, and HPSv2-test distributions, each subsampled to the (p=4, R=5)
fixed-panel design (bootstrap B=50; for HPSv2-test the four strongest
text-to-image generators are held fixed and the COCO real image excluded, so the
anchor reflects generated-only preference).

These are **data** derived from third-party datasets, so they are **fetched
separately** (alongside the TASTE dataset) and are **not committed** to this
repository.

## Getting them

> **TODO (release):** add the download link (Google Drive) here.

Download and place the `.npy` files directly in this directory. The expected
files, for each of `sushi`, `movielens`, `mtbench`, `hpsv2`:

```
<ds>_T.npy         # per-sample mean pairwise Kendall tau
<ds>_pairtau.npy   # per-sample individual pairwise taus
<ds>_pmaj.npy      # per-sample majority-vote probabilities
<ds>_cycle.npy     # per-sample Condorcet-cycle indicator (0/1)
```

plus a few fixed-panel / empirical variants
(`{movielens,mtbench,sushi}_fixed_panel_T.npy`,
`{movielens,mtbench}_empirical_T_p4R5.npy`).

## What uses them

Only the cross-domain figure (`paper_figures/make_paper_figures.py`) and the
anchor overlays in `run_dimension.py` read these arrays. The rest of the TASTE
pipeline (per-dimension statistics, sub-dimension signal, per-rater
agreeableness, hallucination agreement) reproduces **without** them. The anchors
are descriptive visual references on the cross-domain figure, not null
hypotheses.

These arrays were produced by `generate_reference_stats.py`, whose dataset
loaders live in an internal helper package that is not included here.
