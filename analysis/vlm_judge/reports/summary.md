# VLM-as-judge results on TASTE

Per-(model, criterion) accuracy of VLM verdicts against the 5-rater human-majority winner on each pair.  Pairs whose human label is a tie are excluded from the accuracy denominator.

## Overall (macro-averaged across 9 criteria)

| Model | macro acc | macro Spearman | mean latency (s) |
|---|---|---|---|
| Qwen3-VL-4B-flipped | 0.6667 | 0.6 | 0.25 |
| Qwen3-VL-8B-Instruct | 0.6667 | 0.0 | 0.23 |
| HPSv2.1 | 0.5429 | -0.0931 | 0.07 |
| PickScore-v1 | 0.522 | -0.1298 | 0.06 |
| Qwen3-VL-8B-Thinking | 0.5 | -0.2 | 165.68 |
| LAION-Aesthetic-V2 | 0.4995 | -0.1721 | 0.04 |
| Qwen3-VL-4B | 0.3333 | -0.6 | 0.38 |
| Qwen3-VL-8B-Instruct-x5 | 0.3148 | -0.5556 | 2.54 |
| Qwen3-VL-8B-Instruct-single-image | None | None | 0.00 |

## Per-criterion accuracy

| Model | aesthetics_color_harmony | aesthetics_mood | aesthetics_preference | aesthetics_typography | aesthetics_visual_hier | descriptions_color_acc | descriptions_preference | descriptions_spatial_acc | descriptions_typography |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-4B-flipped | -- | -- | -- | -- | -- | -- | -- | -- | 0.667 |
| Qwen3-VL-8B-Instruct | -- | -- | -- | -- | -- | -- | -- | -- | 0.667 |
| HPSv2.1 | 0.571 | 0.590 | 0.561 | 0.504 | 0.487 | 0.542 | 0.577 | 0.523 | 0.531 |
| PickScore-v1 | 0.566 | 0.583 | 0.534 | 0.566 | 0.514 | 0.477 | 0.504 | 0.446 | 0.508 |
| Qwen3-VL-8B-Thinking | -- | -- | -- | -- | -- | -- | -- | -- | 0.500 |
| LAION-Aesthetic-V2 | 0.545 | 0.564 | 0.495 | 0.526 | 0.547 | 0.470 | 0.468 | 0.468 | 0.411 |
| Qwen3-VL-4B | -- | -- | -- | -- | -- | -- | -- | -- | 0.333 |
| Qwen3-VL-8B-Instruct-x5 | 0.500 | 0.500 | 0.167 | 0.333 | 0.333 | 0.167 | 0.167 | 0.333 | 0.333 |
| Qwen3-VL-8B-Instruct-single-image | -- | -- | -- | -- | -- | -- | -- | -- | -- |