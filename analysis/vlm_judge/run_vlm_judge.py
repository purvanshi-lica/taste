"""Drive a Runner over TASTE's pairwise judging tasks.

Reads `pair_jsonl/{criterion}.jsonl` and writes
`results/{model_name}/{criterion}.jsonl` with one verdict per task.
Resumable: skips tasks that already have a result.

Usage:
    python run_vlm_judge.py --runner vllm \
        --model-id Qwen/Qwen2.5-VL-7B-Instruct \
        --criteria descriptions_typography \
        --limit 18                  # smoke test: first 18 pairs only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from prompts.criteria import build_judge_prompt  # noqa: E402
from runners import (  # noqa: E402
    HFTransformersRunner, VLLMRunner, InternVLRunner,
    LAIONAestheticV2Runner, HPSv2_1Runner, PickScoreRunner,
    PairScorerRunner, extract_short_prompt,
)


PAIR_DIR = HERE / "pair_jsonl"
RESULTS_DIR = HERE / "results"


def _load_done(out_path: Path) -> set:
    """Set of (prompt_id, image_a_model, image_b_model) that already
    have a result, so we can resume."""
    done = set()
    if not out_path.exists():
        return done
    with open(out_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                key = (rec["prompt_id"], rec["image_a_model"],
                       rec["image_b_model"])
                done.add(key)
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def run_criterion(runner, criterion: str, limit: int = -1,
                  flip_images: bool = False,
                  prompt_style: str = "reason_then_answer",
                  use_paraphrase_pool: bool = False):
    pair_path = PAIR_DIR / f"{criterion}.jsonl"
    if not pair_path.exists():
        print(f"  [skip] no pair file at {pair_path}")
        return
    out_dir = RESULTS_DIR / runner.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{criterion}.jsonl"
    done = _load_done(out_path)

    n_total = 0
    n_skipped = 0
    n_done_this_run = 0
    n_errors = 0
    t_start = time.time()
    fewshot_excluded = getattr(runner, "_fewshot_pid_excluded", None)

    with open(pair_path) as fin, open(out_path, "a") as fout:
        for line_idx, line in enumerate(fin):
            if 0 <= limit <= line_idx:
                break
            n_total += 1
            task = json.loads(line)
            # Exclude the few-shot example pid from evaluation so we
            # don't grade the model on the same pair we showed it.
            if (fewshot_excluded is not None
                    and task["prompt_id"] == fewshot_excluded):
                n_skipped += 1
                continue
            key = (
                task["prompt_id"],
                task["image_a"]["model"],
                task["image_b"]["model"],
            )
            if key in done:
                n_skipped += 1
                continue

            # Optional: swap A and B before sending to the model.
            # Use only for diagnostics (position-bias check).  When
            # swapping we flip the human_majority too so the resulting
            # verdict-vs-human comparison stays semantically aligned
            # with the original pair identity.
            a_spec = task["image_a"]
            b_spec = task["image_b"]
            human_label = task["human_majority"]
            if flip_images:
                a_spec, b_spec = b_spec, a_spec
                if human_label == "A":
                    human_label = "B"
                elif human_label == "B":
                    human_label = "A"
            try:
                img_a = runner.fetch_image(a_spec["url"])
                img_b = runner.fetch_image(b_spec["url"])
            except Exception as e:
                rec = {
                    "criterion": criterion,
                    "prompt_id": task["prompt_id"],
                    "image_a_model": a_spec["model"],
                    "image_b_model": b_spec["model"],
                    "model_judge": runner.name,
                    "verdict": "unknown",
                    "raw_response": "",
                    "latency_s": 0.0,
                    "ok": False,
                    "error": f"image_fetch: {type(e).__name__}: {e}"[:300],
                    "human_majority": human_label,
                    "flipped": flip_images,
                }
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                n_errors += 1
                continue

            if isinstance(runner, PairScorerRunner):
                # Scoring models: use a noun-phrase summary that
                # matches the T2I-style training distribution of
                # HPSv2 / PickScore.  LAION ignores text anyway.
                prompt = extract_short_prompt(task["prompt"])
                paraphrase_idx = 0
            else:
                # Pick a paraphrase deterministically per (pid, A, B)
                # so the assignment is reproducible across runs.
                if use_paraphrase_pool:
                    paraphrase_idx = (
                        hash((task["prompt_id"],
                              task["image_a"]["model"],
                              task["image_b"]["model"])) % 8
                    )
                else:
                    paraphrase_idx = 0
                prompt = build_judge_prompt(
                    criterion, task["prompt"], style=prompt_style,
                    question_variant=paraphrase_idx,
                )
            res = runner.judge_pair(prompt, img_a, img_b)
            rec = {
                "criterion": criterion,
                "prompt_id": task["prompt_id"],
                "image_a_model": a_spec["model"],
                "image_b_model": b_spec["model"],
                "model_judge": runner.name,
                "verdict": res.verdict,
                "raw_response": res.raw_response,
                "latency_s": round(res.latency_s, 3),
                "ok": res.ok,
                "error": res.error,
                "human_majority": human_label,
                "flipped": flip_images,
                "paraphrase_idx": paraphrase_idx,
            }
            if res.verdicts is not None:
                rec["verdicts"] = res.verdicts
                rec["raw_responses"] = res.raw_responses
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n_done_this_run += 1
            if not res.ok:
                n_errors += 1
            if n_done_this_run % 10 == 0:
                elapsed = time.time() - t_start
                rate = n_done_this_run / max(elapsed, 1e-3)
                print(
                    f"  [{criterion}] done {n_done_this_run} / "
                    f"~{n_total - n_skipped}  rate {rate:.2f} /s",
                    flush=True,
                )

    elapsed = time.time() - t_start
    print(
        f"  [{criterion}] finished: total={n_total} skipped={n_skipped} "
        f"new={n_done_this_run} errors={n_errors}  in {elapsed:.0f}s"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", default="vllm",
                        choices=["vllm", "hf_transformers",
                                 "internvl",
                                 "laion_aesthetic_v2",
                                 "hpsv2_1", "pickscore"])
    parser.add_argument("--model-id", default=None,
                        help="Required for vllm / hf_transformers; "
                             "auto-set for the scorer runners.")
    parser.add_argument("--name", default=None,
                        help="Display name; defaults to last segment of model-id")
    parser.add_argument("--criteria", nargs="+", default=None,
                        help="Specific criterion slugs; default = all 9")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Cap pairs per criterion (smoke testing)")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    parser.add_argument("--max-new-tokens", type=int, default=8,
                        help="Cap on response tokens.  Thinking-mode "
                             "models need 200+ to emit a full CoT.")
    parser.add_argument("--flip-images", action="store_true",
                        help="Send images in swapped order (diagnostics)")
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Number of stochastic samples per pair "
                             "(1 = greedy, deterministic).  When > 1 "
                             "the model is sampled at its recommended "
                             "temperature/top_p/top_k to mimic multiple "
                             "human raters.")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override recommended sampling temperature "
                             "(only used when --num-samples > 1)")
    parser.add_argument("--top-p", type=float, default=None,
                        help="Override recommended top_p")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Override recommended top_k")
    parser.add_argument("--prompt-style", default="reason_then_answer",
                        choices=["single_char", "reason_then_answer",
                                 "reason_with_synthetic_example"],
                        help="single_char: 'A'/'B' only.  "
                             "reason_then_answer: 2-3 sentence rationale "
                             "+ 'Final answer: X' line.  "
                             "reason_with_synthetic_example: same plus a "
                             "per-criterion text-only example.")
    parser.add_argument("--image-layout", default="labeled",
                        choices=["labeled", "unlabeled"],
                        help="`labeled` (default) inserts 'Image A:' / "
                             "'Image B:' text directly before each image "
                             "(field-standard pairwise format).  "
                             "`unlabeled` is the legacy harness format "
                             "where binding is inferred from the text.")
    parser.add_argument("--fewshot-pid", type=int, default=None,
                        help="Real-image few-shot: prompt id whose pair "
                             "becomes the canonical example.  Pair must "
                             "be 5/5-consensus B-winner.  Excluded from "
                             "evaluation.")
    parser.add_argument("--fewshot-criterion", default=None,
                        help="Criterion for the fewshot pair (defaults "
                             "to first --criteria value).")
    parser.add_argument("--fewshot-reasoning", default=None,
                        help="Reasoning text for the fewshot example.  "
                             "Should end 'Final answer: B'.")
    parser.add_argument("--use-paraphrase-pool", action="store_true",
                        help="DistortBench-style: cycle through 8 "
                             "question paraphrases per criterion, "
                             "deterministically by (pid, A_model, "
                             "B_model) hash.  Reduces sensitivity to "
                             "specific phrasing.")
    args = parser.parse_args()

    if args.runner == "vllm":
        if not args.model_id:
            parser.error("--model-id is required for runner=vllm")
        runner = VLLMRunner(
            model_id=args.model_id, name=args.name,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_mem,
            max_new_tokens=args.max_new_tokens,
            num_samples=args.num_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
    elif args.runner == "hf_transformers":
        if not args.model_id:
            parser.error("--model-id is required for runner=hf_transformers")

        # Resolve few-shot example, if requested
        fewshot_example = None
        fewshot_pid_excluded = None
        if args.fewshot_pid is not None:
            fewshot_crit = args.fewshot_criterion or (
                args.criteria[0] if args.criteria else None
            )
            if not fewshot_crit:
                parser.error(
                    "--fewshot-pid requires either --fewshot-criterion or "
                    "--criteria"
                )
            if args.fewshot_reasoning is None:
                parser.error("--fewshot-pid requires --fewshot-reasoning")

            # Find the example pair in pair_jsonl
            ex_path = PAIR_DIR / f"{fewshot_crit}.jsonl"
            example_rec = None
            with open(ex_path) as f:
                for line in f:
                    r = json.loads(line)
                    if r["prompt_id"] == args.fewshot_pid:
                        # Pick the first 5/5 B-winning pair for this pid
                        votes = r.get("human_votes") or {}
                        if (votes.get("B", 0) == 5
                                and votes.get("A", 0) == 0):
                            example_rec = r
                            break
            if example_rec is None:
                parser.error(
                    f"No 5/5 B-winning pair found in {fewshot_crit} "
                    f"for prompt-id {args.fewshot_pid}"
                )

            print(f"loading few-shot example: pid={args.fewshot_pid} "
                  f"criterion={fewshot_crit} "
                  f"A={example_rec['image_a']['model']} "
                  f"B={example_rec['image_b']['model']}")
            from runners import Runner as _Runner
            ex_img_a = _Runner.fetch_image(example_rec["image_a"]["url"])
            ex_img_b = _Runner.fetch_image(example_rec["image_b"]["url"])
            fewshot_example = {
                "image_a": ex_img_a,
                "image_b": ex_img_b,
                "reasoning": args.fewshot_reasoning,
            }
            fewshot_pid_excluded = args.fewshot_pid

        runner = HFTransformersRunner(
            model_id=args.model_id, name=args.name,
            max_new_tokens=args.max_new_tokens,
            num_samples=args.num_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            image_layout=args.image_layout,
            fewshot_example=fewshot_example,
        )
        # Stash the excluded pid on the runner for run_criterion to skip
        runner._fewshot_pid_excluded = fewshot_pid_excluded
    elif args.runner == "internvl":
        if not args.model_id:
            parser.error("--model-id is required for runner=internvl")
        runner = InternVLRunner(
            model_id=args.model_id, name=args.name,
            max_new_tokens=args.max_new_tokens,
            num_samples=args.num_samples,
            image_layout=args.image_layout,
        )
    elif args.runner == "laion_aesthetic_v2":
        runner = LAIONAestheticV2Runner(name=args.name)
    elif args.runner == "hpsv2_1":
        runner = HPSv2_1Runner(name=args.name)
    elif args.runner == "pickscore":
        runner = PickScoreRunner(name=args.name)
    else:
        raise ValueError(f"Unknown runner: {args.runner}")

    print(f"=== {runner.name}  via  {args.runner}  ({args.model_id}) ===")
    runner.warmup()
    print("  warmup ok.  starting eval ...", flush=True)

    if args.criteria is None:
        criteria = sorted(p.stem for p in PAIR_DIR.glob("*.jsonl"))
    else:
        criteria = args.criteria

    for c in criteria:
        run_criterion(runner, c, limit=args.limit,
                      flip_images=args.flip_images,
                      prompt_style=args.prompt_style,
                      use_paraphrase_pool=args.use_paraphrase_pool)

    runner.shutdown()


if __name__ == "__main__":
    main()
