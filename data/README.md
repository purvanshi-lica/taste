# data/

This directory is a placeholder.  The TASTE dataset is **not** committed to
this repository; it is downloaded through a separate interface.

## Privacy

Designer identities are **masked** before release.  The published pairs carry
anonymized rater identifiers (`A1`–`A5` for the Aesthetics cohort, `D1`–`D5`
for the Descriptions cohort) rather than any personal information.  Do not
attempt to de-anonymize raters from the released artifacts.

## Getting the data

> **TODO (release):** fill in the canonical download link before publishing.

The dataset will be available via one of:

- **Hugging Face Hub** — `datasets.load_dataset(...)` (preferred), or
- **Google Drive** — a download archive linked here.

After download, the expected layout for training is:

```
data/
├── battles_train.csv      # training pairs (prompt, image_a, image_b, win_rate / winner, agreement)
├── battles_val.csv        # validation pairs
├── halluc_train.csv       # per-image hallucination labels (optional head)
├── halluc_val.csv
└── images/                # the referenced design images
```

Column semantics for the battle CSVs are documented in
`training/preprocess_data.py` (which produces them from the raw rankings) and
in `training/train.py`'s dataset loader.

## Analysis inputs (masked ranking CSVs)

The analysis framework (`../analysis/`) reads the per-dimension ranking CSVs
from `TASTE_DATA_DIR` (default: this `data/` directory). Ranking files use the
columns:

```
eval_round_stage_id, model, rank, prompt_id, evaluator, prompt, model_output_image_url
```

and the hallucination-flag files use:

```
evaluator, model, hallucination_value, asset_id, prompt_id, hallucination_flag
```

The `evaluator` column carries the masked code (`A1`-`A5` / `D1`-`D5`), never a
real identity.

## Note for inference-only users

You do **not** need this dataset to run the scorer — only a trained
checkpoint (see `../checkpoints/README.md`) and your own image pairs (see
`../examples/input_example.csv`).
