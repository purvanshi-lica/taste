# PROVENANCE

Traceability record for files brought into this repository, so the release
can be mapped back to the source it was assembled from.

## Source

All training-side code and the local reference artifacts were extracted
(read-only, via `git show`) from the internal monorepo branch:

- **repo:** `lica-world/ml-platform`
- **branch:** `hps-contra`
- **commit:** `d0aa1e2`
- **subpath:** `other-projects/hps-contra/`
- **extracted on:** 2026-05-27

The inference package under `src/taste_scorer/` predates this extraction and
was authored as a cleaned, Qwen-only inference wrapper around the same model
architecture.

## What came from where

| destination | source (on `hps-contra@d0aa1e2`) | notes |
|-------------|-----------------------------------|-------|
| `training/train.py`            | `train.py`            | byte-identical |
| `training/heads.py`            | `heads.py`            | byte-identical; functionally identical to `src/taste_scorer/heads.py` (docstring differs) |
| `training/embedders.py`        | `embedders.py`        | full multi-family version (training keeps this; the package version is a Qwen-only rewrite) |
| `training/embed_cache.py`      | `embed_cache.py`      | byte-identical |
| `training/inference.py`        | `inference.py`        | byte-identical; only its eval-metric / HTML-report helpers are used by `train.py` |
| `training/preprocess_data.py`  | `preprocess_data.py`  | byte-identical |
| `training/summarize_sweep.py`  | `summarize_sweep.py`  | byte-identical |
| `training/retrain_best.sh`     | `retrain_best.sh`     | byte-identical |
| `results/01_baseline_sweep.md` | `reports/01_baseline_sweep.md` | local-only (git-ignored) |
| `results/02_tier0_data_diagnostics.md` | `reports/02_tier0_data_diagnostics.md` | local-only (git-ignored) |
| `results/00_onboarding.md`     | `reports/00_onboarding.md` | local-only (git-ignored) |
| `results/premodel.tex`         | `docs/premodel.tex`   | local-only (git-ignored); paper §7 source |

## Intentional duplication

`training/heads.py` and `src/taste_scorer/heads.py` both exist on purpose:
`training/` is the frozen research snapshot that reproduces the published
checkpoint, while `src/taste_scorer/` is the maintained inference package.
They are functionally identical today; do not assume one is generated from
the other.

## Not included

- The trained checkpoint(s) — fetched separately (see `checkpoints/README.md`).
- The dataset — fetched separately and designer-masked (see `data/README.md`).
- `model.py` from the source tree — not needed (the training-side
  `embedders.py` loads the backbone via `transformers` directly).
