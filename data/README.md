# data/

Processing scripts for the **TASTE dataset**, hosted on the Hugging Face Hub:

> [`purvanshi/TASTE`](https://huggingface.co/datasets/purvanshi/TASTE)

The dataset itself is **not** committed to this repository — it is fetched from
the Hub. This directory only holds the scripts that download a local snapshot
and reshape the canonical tables into the artifacts the rest of the repo
consumes:

- per-(track, dimension) **ranking CSVs** → consumed by [`../analysis/`](../analysis/)
- pairwise **battles** (winner-vs-loser) → input format for [`../taste-scorer/`](../taste-scorer/)

## Privacy

Designer identities are **masked** before release: the published rows carry
anonymized evaluator ids (`eval_001`, `eval_002`, …) rather than any personal
information. Do not attempt to de-anonymize raters from the released artifacts.

## Install

```bash
pip install -r requirements.txt
```

## Usage

`process.py` is a small CLI with three subcommands. Run any of them with
`--help` for the full flag list.

```bash
# 1. Snapshot the parquet tables locally (add --with-images for the ~1.6 GB images/)
python process.py download --out raw/

# 2. Write one ranking CSV per (track, dimension) into rankings/
python process.py rankings --raw-dir raw/ --out rankings/

# 3. Derive pairwise battles (winner = image_a, by construction) into battles.csv
python process.py battles --raw-dir raw/ --out battles.csv
```

If you skip step 1, `rankings` and `battles` will stream the parquet tables
directly from the Hub via `hf://` URIs (also requires `huggingface_hub`).

Everything this directory produces (`raw/`, `rankings/`, `battles.csv`, the
`images/` folder, any `*.parquet`) is git-ignored; only the scripts and this
README are tracked.

## The dataset at a glance

The Hub release is a set of canonical, normalized parquet tables (plus two
pre-joined browseable views and the raw `images/`). The scripts here read the
canonical tables:

| Table | One row per | Key columns |
|---|---|---|
| `prompts.parquet` | unique (track, dimension, prompt) | `prompt_id`, `track`, `dimension`, `prompt_id_src`, `prompt_text` |
| `assets.parquet` | generated image | `asset_id`, `model`, `image_url`, `image_path`, `track` |
| `rankings.parquet` | one ranking vote | `eval_round_stage_id`, `dimension`, `track`, `prompt_id`, `asset_id`, `evaluator_id`, `rank` |
| `hallucinations.parquet` | per-image binary judgement | `track`, `prompt_id_src`, `asset_id`, `evaluator_id`, `hallucination_value`, `hallucination_flag` |
| `evaluators.parquet` | anonymized evaluator | `evaluator_id`, `tracks`, `n_ranking_rows`, `n_halluc_rows` |

The corpus is split into two **tracks** — `aesthetics` (does it look good?) and
`descriptions` (does it match the prompt?) — each annotated on its own subset of
ranking **dimensions**. `preference` and `typography` appear in both tracks; the
other five are track-exclusive. See the
[dataset card](https://huggingface.co/datasets/purvanshi/TASTE) for the full
schema and provenance notes.

## Output formats

### Ranking CSVs (`rankings/<slug>.csv`)

One file per (track, dimension), in the long format the analysis framework
reads (see [`../analysis/README.md`](../analysis/README.md)):

```
eval_round_stage_id, model, rank, prompt_id, evaluator, prompt, model_output_image_url
```

The slugs match the analysis dimension keys, e.g. `aesthetics_color_harmony`,
`descriptions_spatial_acc`.

### Battles (`battles.csv`)

Every ranked group is expanded into ordered pairs where `image_a` outranks
`image_b` (so `image_a` is the preferred design). Columns:

```
pair_id, track, dimension, prompt, prompt_id,
image_a, image_b, model_a, model_b,
asset_id_a, asset_id_b, evaluator_id, rank_a, rank_b, winner
```

The `prompt`, `image_a`, `image_b`, `model_a`, `model_b` columns line up with
the scorer's input CSV format ([`../taste-scorer/README.md`](../taste-scorer/README.md)).

## Note for inference-only users

You do **not** need this dataset to run the scorer — only a trained checkpoint
and your own image pairs. See [`../taste-scorer/README.md`](../taste-scorer/README.md).
