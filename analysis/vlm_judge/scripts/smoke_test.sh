#!/bin/bash
# Smoke test: run the smallest-available open VLM on the first 6 pairs
# of one criterion (Descriptions Typography, the strongest-signal one,
# so the result is interpretable).  This validates: pair-jsonl
# extraction, image fetch, prompt format, model load, response
# parsing, and result-file plumbing.
set -e

cd "$(dirname "$0")/.."

CRITERION="${CRITERION:-descriptions_typography}"
LIMIT="${LIMIT:-6}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}"
RUNNER="${RUNNER:-hf_transformers}"

echo "=== smoke test ==="
echo "  criterion = $CRITERION"
echo "  limit     = $LIMIT pairs"
echo "  model     = $MODEL_ID"
echo "  runner    = $RUNNER"
echo

# 1. ensure pair jsonl exists
if [ ! -f "pair_jsonl/${CRITERION}.jsonl" ]; then
    echo "(building pair jsonls)"
    python prepare_pairs.py
fi

# 2. run the model
python run_vlm_judge.py \
    --runner "$RUNNER" \
    --model-id "$MODEL_ID" \
    --criteria "$CRITERION" \
    --limit "$LIMIT"

# 3. summarise
python analyze_results.py
echo
echo "=== smoke summary ==="
cat reports/summary.md | head -30 || true
