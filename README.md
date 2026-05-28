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

## Install

```bash
git clone <this repo>
cd taste-scorer
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

Download a published checkpoint and point `--checkpoint` at the extracted
directory.  See [`checkpoints/README.md`](checkpoints/README.md) for the
download link and the exact format, or train your own (see
[Training](#training)).

## Input CSV format

The scorer reads a CSV with three required columns (plus one optional
passthrough):

| column     | required | description                                                       |
|------------|----------|-------------------------------------------------------------------|
| `prompt`   | yes      | The text prompt the two images are being compared against.        |
| `image_a`  | yes      | Path to image A.  Absolute, or relative to `--image-dir`.         |
| `image_b`  | yes      | Path to image B.  Absolute, or relative to `--image-dir`.         |
| `pair_id`  | no       | Optional user-side identifier; passed through to the output CSV.  |

See [`examples/input_example.csv`](examples/input_example.csv) for a
ready-to-edit template.

```csv
pair_id,prompt,image_a,image_b
1,"A clean coffee shop poster with bold typography",poster_a_001.jpg,poster_b_001.jpg
2,"Minimalist book cover, sans-serif title",cover_a_002.jpg,cover_b_002.jpg
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

## Use as a Python library

```python
import pandas as pd
from taste_scorer import PreferenceScorer

scorer = PreferenceScorer.from_checkpoint(
    "path/to/best_pairwise/best",
    device="cuda",  # optional; auto-detected
)
print(scorer.dimensions)        # ['color_accuracy', 'color_harmony', ...]
print(scorer.has_halluc_head)   # True / False

df_in  = pd.read_csv("input.csv")
df_out = scorer.score_pairs(df_in, image_dir="path/to/images/")
df_out.to_csv("scored.csv", index=False)
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

## Training

The inference package above runs a *trained* checkpoint.  To reproduce the
model from the paper (the pairwise-difference head, val accuracy ≈ 0.611) or
to train your own, see [`training/`](training/):

```bash
cd training
bash retrain_best.sh        # needs the dataset under ../data/ and a GPU
```

`training/` is a self-contained snapshot of the original training code; full
details, the architecture, and the expected results are in
[`training/README.md`](training/README.md).  The dataset is fetched separately
and designer-masked — see [`data/README.md`](data/README.md).

## Project layout

```
taste-scorer/
├── README.md
├── LICENSE
├── CITATION.cff
├── PROVENANCE.md                # where bundled files came from
├── pyproject.toml
├── requirements.txt
├── data/                        # placeholder — dataset fetched separately (masked)
│   └── README.md
├── checkpoints/                 # placeholder — checkpoints fetched separately
│   └── README.md
├── examples/
│   └── input_example.csv
├── results/                     # local reference only (git-ignored)
│   └── README.md
├── scripts/
│   └── score.py                 # thin CLI wrapper (no install needed)
├── tests/
│   └── test_smoke.py            # checkpoint-free smoke tests
├── training/                    # research code to reproduce the model
│   ├── README.md
│   ├── train.py · retrain_best.sh
│   ├── heads.py · embedders.py · embed_cache.py · inference.py
│   └── preprocess_data.py · summarize_sweep.py
└── src/taste_scorer/            # the pip-installable inference package
    ├── __init__.py              # public re-exports
    ├── model.py                 # Qwen3-VL embedding wrapper
    ├── embedders.py             # VLEmbedder + QwenVLEmbedder (Qwen-only)
    ├── heads.py                 # PairwiseMultiHeadScorer + HallucinationHead
    ├── scorer.py                # PreferenceScorer (public API)
    └── cli.py                   # taste-score entry point
```

## What the model is and isn't

It is a *modular preference head* trained to imitate a panel of human
annotators on design-quality comparisons.  It is **not** an
image-generation reward model in the HPS/HPSv2 sense (it does not score
absolute quality of a single image); it answers "given this prompt,
which of these two designs would the panel prefer along dimension *d*?"

It is **not** a hallucination detector beyond the binary head shipped
with the checkpoint, which scores per-image rendering-error probability
and is intentionally biased toward higher recall than precision.

## Citing

If you use TASTE-scorer or the TASTE dataset, please cite the paper:
[arXiv:2605.20731](https://arxiv.org/abs/2605.20731) (see
[`CITATION.cff`](CITATION.cff) for the machine-readable entry).

## License

MIT — see [LICENSE](LICENSE).
