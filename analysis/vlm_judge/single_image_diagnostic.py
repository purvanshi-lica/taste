"""Single-image diagnostic for the position-bias collapse finding.

Hypothesis: when both images are passed in the same turn, the VLM's
verdict collapses to "always A".  If we present each image in
isolation and ask for a Likert 1-5 rating, the resulting per-image
ranking (a) shouldn't have a "position" to be biased toward, and (b)
gives us a clean alternative pairwise prediction we can compare against
the side-by-side run.

Outcome interpretations:

  * Per-image scores produce a *different* pairwise ordering than the
    side-by-side run → side-by-side input is the mechanism (Tian et
    al. story).
  * Per-image scores are degenerate (e.g. all 5/5) → model can't
    discriminate at all on this task; position bias is a symptom of
    that.
  * Per-image scores match human majority closely → side-by-side
    bias is recoverable just by changing the input format.

This script is intentionally narrow: one criterion, the same 6-pair
smoke set we used for the pairwise side-by-side run, no batching, no
sampling.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from prompts.criteria import CRITERIA  # noqa: E402
from runners.base import Runner  # noqa: E402
from runners.hf_local import HFTransformersRunner  # noqa: E402


CRITERION = "descriptions_typography"
LIMIT_PAIRS = 6
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
NAME = "Qwen3-VL-8B-Instruct-single-image"
MAX_NEW_TOKENS = 4


def build_single_image_prompt(criterion: str, design_brief: str) -> str:
    spec = CRITERIA[criterion]
    return (
        f"You are a professional graphic designer evaluating one AI-"
        f"generated image for the brief below.\n\n"
        f"BRIEF:\n{design_brief}\n\n"
        f"CRITERION: {spec['display']}\n"
        f"RUBRIC: {spec['rubric']}\n\n"
        f"You see one image.  {spec['question']}\n\n"
        f"Rate on a scale of 1 (very poor) to 5 (excellent).  Reply "
        f"with exactly one digit (1, 2, 3, 4, or 5).  Do not include "
        f"any other text."
    )


def parse_likert(text: str) -> int | None:
    if not text:
        return None
    m = re.search(r"\b([1-5])\b", text)
    if m:
        return int(m.group(1))
    return None


class SingleImageRunner(HFTransformersRunner):
    """Rate a single image Likert 1-5 in isolation."""

    def _build_messages(self, prompt_text, image_a, image_b=None):
        # Override the base's two-image template with a one-image one.
        return [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_a},
                {"type": "text", "text": prompt_text},
            ],
        }]

    def score_single(self, prompt_text, image):
        import torch
        if self._model is None:
            self.warmup()
        messages = self._build_messages(prompt_text, image)
        inputs = self._processor.apply_chat_template(
            messages, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt",
        )
        inputs = {
            k: v.to(self._model.device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }
        with torch.no_grad():
            output = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=False, temperature=None, top_p=None, top_k=None,
            )
        in_len = inputs["input_ids"].shape[1]
        gen = output[0, in_len:]
        text = self._processor.tokenizer.decode(gen, skip_special_tokens=True)
        return parse_likert(text), text


def main():
    pair_path = HERE / "pair_jsonl" / f"{CRITERION}.jsonl"
    pairs = []
    with open(pair_path) as f:
        for i, line in enumerate(f):
            if i >= LIMIT_PAIRS:
                break
            pairs.append(json.loads(line))
    print(f"Loaded {len(pairs)} pairs from {CRITERION}")

    # Collect unique (prompt_id, model_name, image_url) tuples
    unique_imgs = {}
    for p in pairs:
        for side in ("image_a", "image_b"):
            spec = p[side]
            key = (p["prompt_id"], spec["model"])
            unique_imgs[key] = spec["url"]
    print(f"Unique images to score: {len(unique_imgs)}")

    runner = SingleImageRunner(
        model_id=MODEL_ID, name=NAME, max_new_tokens=MAX_NEW_TOKENS,
    )
    runner.warmup()
    print("warmup ok\n")

    # Score each image once
    scores = {}
    raws = {}
    text_prompt = build_single_image_prompt(CRITERION, pairs[0]["prompt"])
    for (pid, model), url in unique_imgs.items():
        t0 = time.time()
        try:
            img = runner.fetch_image(url)
        except Exception as e:
            print(f"  fetch FAIL pid={pid} model={model}: {e}")
            continue
        score, raw = runner.score_single(text_prompt, img)
        elapsed = time.time() - t0
        scores[(pid, model)] = score
        raws[(pid, model)] = raw
        print(f"  pid={pid} model={model:<25} likert={score}  raw={raw!r:<10}"
              f"  ({elapsed:.1f}s)")

    print()
    print("=" * 80)
    print(f"Single-image scores summary:")
    print("=" * 80)
    by_model = defaultdict(list)
    for (pid, m), s in scores.items():
        if s is not None:
            by_model[m].append(s)
    for m, vals in sorted(by_model.items()):
        print(f"  {m:<28}  scores={vals}  mean={sum(vals)/len(vals):.2f}")
    print()

    # Reconstruct pairwise predictions
    n_a = n_b = n_tie = 0
    n_correct = 0
    n_eligible = 0
    print("=" * 80)
    print("Pairwise predictions reconstructed from per-image Likert scores:")
    print("=" * 80)
    print(f"{'pid':<6} {'A model':<22} {'B model':<22} {'A':<3} {'B':<3} "
          f"{'pred':<5} {'human':<5} {'match'}")
    for p in pairs:
        a_key = (p["prompt_id"], p["image_a"]["model"])
        b_key = (p["prompt_id"], p["image_b"]["model"])
        a_s = scores.get(a_key)
        b_s = scores.get(b_key)
        if a_s is None or b_s is None:
            verdict = "?"
        elif a_s > b_s:
            verdict = "A"
            n_a += 1
        elif b_s > a_s:
            verdict = "B"
            n_b += 1
        else:
            verdict = "tie"
            n_tie += 1
        human = p["human_majority"]
        match = ""
        if verdict in ("A", "B") and human in ("A", "B"):
            n_eligible += 1
            if verdict == human:
                n_correct += 1
                match = "OK"
            else:
                match = "MISS"
        print(
            f"{p['prompt_id']:<6} "
            f"{p['image_a']['model'][:21]:<22} "
            f"{p['image_b']['model'][:21]:<22} "
            f"{a_s!s:<3} {b_s!s:<3} "
            f"{verdict:<5} {human:<5} {match}"
        )

    print()
    print(f"Vote spread: A={n_a}  B={n_b}  tie={n_tie}")
    print(f"Accuracy vs human-majority (eligible): "
          f"{n_correct}/{n_eligible}")

    # Side-by-side comparison
    sxs_path = HERE / "results" / "Qwen3-VL-8B-Instruct" / f"{CRITERION}.jsonl"
    if sxs_path.exists():
        sxs_recs = [json.loads(l) for l in open(sxs_path)][:LIMIT_PAIRS]
        sxs_a = sum(1 for r in sxs_recs if r["verdict"] == "A")
        sxs_b = sum(1 for r in sxs_recs if r["verdict"] == "B")
        sxs_correct = sum(
            1 for r in sxs_recs
            if r["verdict"] in ("A", "B")
            and r["verdict"] == r["human_majority"]
        )
        print()
        print("=" * 80)
        print("Comparison vs side-by-side (same model, same 6 pairs):")
        print("=" * 80)
        print(f"  side-by-side  : A={sxs_a}  B={sxs_b}  correct={sxs_correct}/{LIMIT_PAIRS}")
        print(f"  single-image  : A={n_a}  B={n_b}  correct={n_correct}/{n_eligible}")

    # Persist for the paper
    out = HERE / "results" / NAME / f"{CRITERION}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "criterion": CRITERION,
        "model_id": MODEL_ID,
        "n_pairs": len(pairs),
        "n_unique_images": len(unique_imgs),
        "per_image_scores": [
            {"prompt_id": pid, "model": m, "likert": s, "raw": raws.get((pid, m))}
            for (pid, m), s in scores.items()
        ],
        "vote_spread": {"A": n_a, "B": n_b, "tie": n_tie},
        "accuracy_vs_human": {
            "correct": n_correct, "eligible": n_eligible,
        },
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
