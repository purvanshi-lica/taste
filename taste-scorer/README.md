# TASTE-scorer

Inference for the **TASTE preference model**: given an image pair and a
text prompt, the scorer returns a per-dimension probability that
*image A* is preferred over *image B*, plus an optional per-image
hallucination probability.

The model is a small modular preference head trained on top of a frozen
[Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)
backbone, with one MLP per ranking dimension and a *pairwise-difference*
fusion: each per-dimension head consumes
`[t, i_a, i_b, i_a − i_b, |i_a − i_b|, t ⊙ (i_a − i_b)]` and emits a
Bradley-Terry logit directly.

This package supports only the **pairwise-head** architecture (the
best-performing variant in the training repository).

## TL;DR

1. **Install + get the checkpoint** — `pip install -e .`, then download and
   unzip [`TASTE_Checkpoint.zip`](https://storage.googleapis.com/lica-assets/TASTE/TASTE_Checkpoint.zip).
2. **Make an input CSV** — one row per image pair, with the columns
   `prompt, image_a, image_b` (add `model_a, model_b` if you want a leaderboard).
3. **Score it** — `taste-score score input.csv --output-csv scored.csv --checkpoint <dir> --image-dir <imgs>`.
4. **(Optional) Aggregate** — `taste-score leaderboard scored.csv -o leaderboard.csv`.

**Outputs:** a scored CSV (`scored.csv`, your rows plus a `prob_a_wins_<dim>`
column per dimension), and — if you ran step 4 — a `leaderboard.csv` plus a
per-model ranking table printed to the console.

See the sections below for the exact CSV schemas, flags, and the Python API.

## Install

```bash
git clone https://github.com/purvanshi-lica/taste.git
cd taste/taste-scorer
pip install -e .
```

This pulls in `torch`, `transformers`, `qwen-vl-utils`, `peft`, and a
small set of utilities (see `requirements.txt` for the full list).

A CUDA-capable GPU is recommended.

## Get a checkpoint

The checkpoint is a directory containing four files:

```
checkpoint_dir/
├── heads.pt           # PairwiseMultiHeadScorer state dict
├── halluc_head.pt     # HallucinationHead state dict (optional)
├── meta.json          # backbone name, dimensions, logit_scale, etc.
└── lora_adapter/      # PEFT LoRA adapter (optional, only if LoRA-trained)
```

Download the published checkpoint and unzip it, then point `--checkpoint` at the
extracted directory:

```bash
curl -L -o TASTE_Checkpoint.zip \
    https://storage.googleapis.com/lica-assets/TASTE/TASTE_Checkpoint.zip
unzip TASTE_Checkpoint.zip
```

## Input CSV format

The scorer reads a CSV with three required columns plus three optional
passthroughs:

| column     | required | description                                                                                |
|------------|----------|--------------------------------------------------------------------------------------------|
| `prompt`   | yes      | The text prompt the two images are being compared against.                                 |
| `image_a`  | yes      | Path to image A.  Absolute, or relative to `--image-dir`.                                  |
| `image_b`  | yes      | Path to image B.  Absolute, or relative to `--image-dir`.                                  |
| `pair_id`  | no       | Optional user-side identifier; passed through to the output CSV.                           |
| `model_a`  | no       | Generator name for `image_a`.  Pass through; required to build a leaderboard.              |
| `model_b`  | no       | Generator name for `image_b`.  Pass through; required to build a leaderboard.              |

See [`examples/input_example.csv`](examples/input_example.csv) for a
ready-to-edit template.

```csv
pair_id,prompt,image_a,image_b,model_a,model_b
1,"A clean coffee shop poster with bold typography",poster_a_001.jpg,poster_b_001.jpg,gpt-image-1.5,flux-2-max
2,"Minimalist book cover, sans-serif title",cover_a_002.jpg,cover_b_002.jpg,gpt-image-1.5,nano-banana-2
```

Images are validated on disk before any model forward pass; missing
files raise `FileNotFoundError` immediately.

## Output CSV format

The output CSV contains the input columns plus, for every ranking
dimension the head was trained on, a column `prob_a_wins_<dim>` in
`[0, 1]` — the calibrated probability that A is preferred over B on
that dimension.  If the checkpoint has a hallucination head, two
additional columns `halluc_prob_a` and `halluc_prob_b` (also in
`[0, 1]`) hold the per-image hallucination probabilities.

Example output for a checkpoint with the seven design dimensions and a
halluc head:

```csv
pair_id,prompt,image_a,image_b,prob_a_wins_color_accuracy,prob_a_wins_color_harmony,prob_a_wins_mood_and_color_tone,prob_a_wins_preference,prob_a_wins_spatial_accuracy,prob_a_wins_typography,prob_a_wins_visual_hierarchy,halluc_prob_a,halluc_prob_b
1,...,...,...,0.62,0.55,0.71,0.63,0.49,0.78,0.66,0.05,0.12
```

`prob_a_wins_<dim> > 0.5` means the model thinks image A is preferred on
that dimension; closer to 0.5 means the model is uncertain.

## Run from the command line

After `pip install -e .`:

```bash
taste-score score input.csv \
    --output-csv scored.csv \
    --checkpoint /path/to/best_pairwise/best \
    --image-dir   /path/to/images/
```

Without installing, use the script directly:

```bash
python scripts/score.py score input.csv \
    --output-csv scored.csv \
    --checkpoint /path/to/best_pairwise/best \
    --image-dir   /path/to/images/
```

Flags:

| flag             | description                                                                                    |
|------------------|------------------------------------------------------------------------------------------------|
| `--checkpoint`   | Path to the checkpoint directory (required).                                                   |
| `--output-csv`   | Where to write the scored CSV (required).                                                      |
| `--image-dir`    | Root for relative image paths in the input CSV.                                                |
| `--device`       | Torch device (`cuda` / `cpu` / `mps`).  Auto-detected if omitted.                              |
| `--batch-size`   | Head-forward batch size; default 64.  Image / text encoding is one forward per unique value.   |
| `--leaderboard`  | Optional path; if set and the input CSV has `model_a` / `model_b`, also write a leaderboard.   |

## Per-model leaderboard

When the input CSV labels each pair with the generator that produced
each image (`model_a`, `model_b`), the scored CSV carries those columns
through unchanged and you can aggregate them into a ranking.  Every
battle contributes one win-probability sample for `model_a`
(= `prob_a_wins_<dim>`) and one for `model_b` (= `1 − prob_a_wins_<dim>`),
so each model is evaluated under the same number of pairs.

In one shot from the CLI:

```bash
taste-score score input.csv \
    --output-csv scored.csv \
    --leaderboard leaderboard.csv \
    --checkpoint /path/to/best_pairwise/best \
    --image-dir   /path/to/images/
```

Or post-hoc from an already-scored CSV:

```bash
taste-score leaderboard scored.csv -o leaderboard.csv
```

The leaderboard has one row per generator, one column per ranking
dimension, plus `overall` (mean across dimensions), `n_pairs` (sample
size), and — when the checkpoint has a halluc head — `halluc_rate`
(mean predicted hallucination probability across appearances).  Models
are sorted by `overall` descending.

```
                      color_accuracy  color_harmony  ...  overall  n_pairs  halluc_rate
model
gpt-image-1.5                 0.612          0.598  ...    0.604      512        0.061
flux-2-max                    0.554          0.537  ...    0.541      498        0.094
nano-banana-2                 0.488          0.501  ...    0.493      503        0.118
seedream-5-lite               0.346          0.364  ...    0.362      521        0.227
```

## Use as a Python library

```python
import pandas as pd
from taste_scorer import PreferenceScorer, compute_leaderboard, format_leaderboard

scorer = PreferenceScorer.from_checkpoint(
    "path/to/best_pairwise/best",
    device="cuda",  # optional; auto-detected
)
print(scorer.dimensions)        # ['color_accuracy', 'color_harmony', ...]
print(scorer.has_halluc_head)   # True / False

df_in  = pd.read_csv("input.csv")
df_out = scorer.score_pairs(df_in, image_dir="path/to/images/")
df_out.to_csv("scored.csv", index=False)

# If df_in had model_a / model_b, df_out has them too — build the leaderboard:
leaderboard = compute_leaderboard(df_out)
print(format_leaderboard(leaderboard))
leaderboard.to_csv("leaderboard.csv")
```

`score_pairs` deduplicates: each unique prompt and each unique image
path is encoded by the backbone exactly once, regardless of how many
input rows reference it.  The dominant cost is therefore the number of
distinct prompts / images, not the number of pairs.

## Ranking N candidates (best-of-N)

The trained model is pairwise by construction, so there is no
"score one image in isolation" mode.  To rank N candidates against a
prompt:

1. Build an input CSV with one row per ordered pair `(i, j)`, `i ≠ j`,
   sharing the same `prompt` column.
2. Run `taste-score score`.
3. Aggregate the resulting `prob_a_wins_<dim>` matrix into a ranking
   per dimension — counting wins (`prob > 0.5`), or fitting a
   Bradley-Terry score in closed form from the win-probability matrix.

For three candidates `c1, c2, c3` and the `preference` dimension, the
total number of head forwards is 6 (the 3×3 minus-diagonal grid), all
batched in a single call.

## Project layout

```
taste-scorer/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── examples/
│   └── input_example.csv
├── scripts/
│   └── score.py                 # thin CLI wrapper (no install needed)
└── src/taste_scorer/
    ├── __init__.py              # public re-exports
    ├── model.py                 # Qwen3-VL embedding wrapper
    ├── embedders.py             # VLEmbedder + QwenVLEmbedder
    ├── heads.py                 # PairwiseMultiHeadScorer + HallucinationHead
    ├── scorer.py                # PreferenceScorer (public API)
    └── cli.py                   # taste-score entry point
```

## License

MIT — see [LICENSE](LICENSE).
