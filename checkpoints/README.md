# checkpoints/

This directory is a placeholder.  Trained checkpoints are **not** committed;
they are downloaded separately.

## Checkpoint format

A checkpoint is a directory with:

```
<name>/
├── heads.pt           # PairwiseMultiHeadScorer state dict
├── halluc_head.pt     # HallucinationHead state dict (optional)
├── meta.json          # backbone name, dimensions, logit_scale, etc.
└── lora_adapter/      # PEFT LoRA adapter (optional; only if LoRA-trained)
```

This is exactly the format produced by `training/retrain_best.sh` and consumed
by `taste_scorer.PreferenceScorer.from_checkpoint`.

## Getting a checkpoint

> **TODO (release):** fill in the canonical checkpoint link before publishing
> (Hugging Face model hub entry, matching the data-hosting choice).

Once downloaded and extracted here, point the scorer at it:

```bash
taste-score score input.csv \
    --output-csv scored.csv \
    --checkpoint checkpoints/best_pairwise/best \
    --image-dir  /path/to/images/
```

## Producing your own

See `../training/README.md`.  `training/retrain_best.sh` reproduces the
paper's best pairwise-difference head and writes its checkpoint into
`checkpoints/best_pairwise/best/`.
