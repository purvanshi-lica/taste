# training/

Code to train the TASTE preference model (the per-dimension pairwise-difference
head on a frozen Qwen3-VL-Embedding-2B backbone).  This is the research code
that produced the checkpoint the `taste_scorer` inference package loads.

> **Self-contained by design.**  The modules here (`heads.py`, `embedders.py`,
> `inference.py`, `embed_cache.py`) are the *original training-time* sources,
> kept byte-identical to the run that produced the published model so results
> reproduce exactly.  The inference package under `../src/taste_scorer/` is a
> trimmed extract of the same architecture (Qwen-only embedder, inference-only
> heads).  The two `heads.py` are functionally identical; `embedders.py`
> differs (training keeps the full multi-family version).  Run training from
> inside this directory so the local modules import correctly.

## Architecture

```
text, image ──Qwen3-VL (frozen)──▶ text_emb, image_emb
                                        │
                                        ├─▶ per-dimension pairwise MLP  (Bradley-Terry logit)
                                        └─▶ hallucination head          (binary BCE)
```

The pairwise head consumes the fused
`[t, i_a, i_b, i_a − i_b, |i_a − i_b|, t ⊙ (i_a − i_b)]` and emits the
Bradley-Terry logit directly.  The backbone is frozen by default; pass
`--enable-lora` to additionally fit a LoRA adapter (this is the §7 LoRA
variant, which scored *below* the frozen head).

## Reproduce the paper's model

```bash
# from this directory, with the dataset under ../data/ (see ../data/README.md)
bash retrain_best.sh
```

`retrain_best.sh` runs the best configuration from the sweep (config H:
pairwise-difference head, hidden dim 128, dropout 0.2, weight decay 0.05,
per-evaluator hard labels with agreement weighting, no criterion-conditioned
prompt).  Expected validation results:

| metric | value |
|--------|-------|
| overall val accuracy | ≈ 0.611 |
| unanimous (5-0) / majority (4-1) / split (3-2) | ≈ 0.65 / 0.60 / 0.60 |
| best epoch | ~150 |

The checkpoint is written to `../checkpoints/best_pairwise/best/`
(`heads.pt`, `halluc_head.pt`, `meta.json`) — the format
`taste_scorer.PreferenceScorer.from_checkpoint` expects.

## Files

| file | purpose |
|------|---------|
| `train.py`            | Training entry point (BT loss + halluc BCE, agreement weighting, early stopping, sweep flags). |
| `retrain_best.sh`     | One-command reproduction of the best (0.611) model. |
| `heads.py`            | Model heads (scalar + pairwise MLP, hallucination head). |
| `embedders.py`        | Frozen-backbone embedder (`VLEmbedder.from_pretrained`; multi-family). |
| `embed_cache.py`      | Caches frozen-backbone embeddings (50–100× speedup for MLP-only training). |
| `inference.py`        | Eval metrics + HTML report helpers used during validation. |
| `preprocess_data.py`  | Builds the battle CSVs from the raw designer rankings. |
| `summarize_sweep.py`  | Aggregates a hyper-parameter sweep into a comparison table. |

## Requirements

Training needs `torch`, `transformers`, `peft`, `accelerate`, `qwen-vl-utils`
(same stack as the inference package; see `../requirements.txt`) plus a
CUDA-capable GPU.  `wandb` is optional (pass `--wandb-project`).
