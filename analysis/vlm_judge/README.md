# VLM-as-judge harness for TASTE

Drives a small slate of vision-language models through TASTE's
pairwise design-comparison task and reports per-criterion accuracy
against human-majority targets.  Mirrors the model slate in
DistortBench~\cite{Goyal2026DistortBench} where feasible.

## Pipeline overview

```
csv/                                   ->  pair_jsonl/{criterion}.jsonl
  raw TASTE rankings                       (4320 pairs total = 9 criteria
                                              x 80 prompts x 6 pairs)

pair_jsonl/{criterion}.jsonl           ->  results/{model}/{criterion}.jsonl
  pairs + ground-truth majority             VLM verdicts per pair

results/{model}/{criterion}.jsonl      ->  reports/summary.json
  raw verdicts                              accuracy + Spearman per
                                              (model, criterion)
```

## Files

- `prepare_pairs.py`  -- extract per-criterion pairwise tasks from CSVs
- `prompts/criteria.py` -- per-criterion rubric templates
- `runners/base.py` -- abstract `Runner` interface
- `runners/hf_local.py` -- HuggingFace transformers / vLLM runner
- `runners/api.py` -- API runner (TODO, deferred)
- `run_vlm_judge.py` -- main eval driver
- `analyze_results.py` -- aggregate and report
- `scripts/smoke_test.sh` -- 3-prompt × 1-criterion check

## Model slate (target)

Mirrors DistortBench Table 2.

| Family | Sizes | Source |
|---|---|---|
| Qwen3.5-VL | 4B, 9B, 27B, 35B (+ Think variants) | Qwen/Qwen3.5-VL-* |
| Gemma 3 | 4B, 12B, 27B | google/gemma-3-* |
| InternVL3.5 | 8B, 14B, 38B | OpenGVLab/InternVL3_5-* |
| Kimi-VL | A3B (+ Think) | moonshotai/Kimi-VL-A3B-Instruct |
| Qwen2-VL | 7B (legacy reference) | Qwen/Qwen2-VL-7B-Instruct |
| GPT-5 / Claude Opus 4.7 | -- | API (deferred) |

Smoke test starts with smallest open model (Qwen3.5-VL-4B if
available, falling back to InternVL3.5-8B).

## Compute notes

- Single B200 (183GB VRAM) on the TASTE workstation
- Up to 38B models fit comfortably in BF16; 70B-class needs INT8 quant
- vLLM (env: `vllm-eval`) preferred for throughput; transformers
  fallback when vLLM lacks support
- ~4320 pair calls per model; at 1-2 calls/s on a B200 that is
  roughly 1-2 hours per local model run

## Inputs and outputs

Each pair JSONL line:
```json
{
  "criterion": "aesthetics_typography",
  "prompt_id": 608,
  "prompt": "<full structured prompt text>",
  "image_a": {"model": "FLUX.2 [max]", "url": "https://..."},
  "image_b": {"model": "Nano Banana 2", "url": "https://..."},
  "human_majority": "B",
  "human_votes": {"A": 1, "B": 4},
  "human_tie": false
}
```

Each result JSONL line (per model per pair):
```json
{
  "criterion": "...", "prompt_id": 608,
  "model_judge": "Qwen3.5-VL-9B",
  "verdict": "B", "raw_response": "...",
  "latency_s": 1.2, "ok": true
}
```
