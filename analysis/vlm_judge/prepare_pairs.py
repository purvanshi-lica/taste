"""Build per-criterion JSONL files of pairwise judging tasks.

For each TASTE criterion (9 total) and each prompt (80 per criterion),
enumerate the C(4,2)=6 (image_a, image_b) pairs of generator outputs.
Compute the human-majority verdict from the 5-rater rankings: image A
wins if more raters rank it above image B than vice versa, tie if
equal.

Outputs `pair_jsonl/{criterion_slug}.jsonl` with one task per line.
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DT = HERE.parent / "distribution_tests"
sys.path.insert(0, str(DT))
from taste_stats import DIMENSIONS, DATA_DIR, MODELS  # noqa: E402

OUT_DIR = HERE / "pair_jsonl"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_dim(slug: str, csv_name: str):
    """Load one criterion CSV and return a per-(prompt, evaluator)
    rank_map dict plus per-(prompt, model) image-URL map."""
    df = pd.read_csv(DATA_DIR / csv_name)
    df = df.drop_duplicates(subset=["model", "rank", "prompt_id", "evaluator"])
    # Validate strict 4-way per (prompt, eval)
    valid = df.groupby(["prompt_id", "evaluator"]).filter(
        lambda x: sorted(x["rank"].tolist()) == [1, 2, 3, 4]
    )
    rank_map = {}      # (prompt_id, evaluator) -> {model: rank}
    img_url = {}       # (prompt_id, model) -> url
    prompt_text = {}   # prompt_id -> prompt text
    for _, row in valid.iterrows():
        pid = int(row["prompt_id"])
        ev = row["evaluator"]
        m = row["model"]
        rank_map.setdefault((pid, ev), {})[m] = int(row["rank"])
        img_url[(pid, m)] = str(row["model_output_image_url"])
        if pid not in prompt_text:
            prompt_text[pid] = str(row["prompt"])
    return rank_map, img_url, prompt_text


def _majority_verdict(rank_map, prompt_id, evaluators, model_a, model_b):
    """Return ('A'|'B'|'tie', vote_count_for_A, vote_count_for_B)."""
    a_wins = 0
    b_wins = 0
    n_total = 0
    for ev in evaluators:
        ranks = rank_map.get((prompt_id, ev))
        if ranks is None:
            continue
        if model_a not in ranks or model_b not in ranks:
            continue
        n_total += 1
        if ranks[model_a] < ranks[model_b]:
            a_wins += 1
        elif ranks[model_b] < ranks[model_a]:
            b_wins += 1
    if a_wins > b_wins:
        return "A", a_wins, b_wins
    elif b_wins > a_wins:
        return "B", a_wins, b_wins
    return "tie", a_wins, b_wins


def build_one(slug: str, display: str, group: str, csv_name: str):
    rank_map, img_url, prompt_text = _load_dim(slug, csv_name)
    prompt_ids = sorted({pid for (pid, _) in rank_map.keys()})
    evaluators = sorted({ev for (_, ev) in rank_map.keys()})

    out_path = OUT_DIR / f"{slug}.jsonl"
    n_tasks = 0
    with open(out_path, "w") as f:
        for pid in prompt_ids:
            for model_a, model_b in combinations(MODELS, 2):
                verdict, a_votes, b_votes = _majority_verdict(
                    rank_map, pid, evaluators, model_a, model_b
                )
                task = {
                    "criterion": slug,
                    "criterion_display": display,
                    "criterion_group": group,
                    "prompt_id": pid,
                    "prompt": prompt_text[pid],
                    "image_a": {
                        "model": model_a,
                        "url": img_url[(pid, model_a)],
                    },
                    "image_b": {
                        "model": model_b,
                        "url": img_url[(pid, model_b)],
                    },
                    "human_majority": verdict,
                    "human_votes": {"A": a_votes, "B": b_votes},
                }
                f.write(json.dumps(task) + "\n")
                n_tasks += 1
    print(f"  {slug:30s}  {n_tasks:>4d} pairs  ->  {out_path}")
    return n_tasks


def main():
    total = 0
    for slug, display, group, csv_name in DIMENSIONS:
        total += build_one(slug, display, group, csv_name)
    print(f"\nTotal pair-tasks across {len(DIMENSIONS)} criteria: {total}")


if __name__ == "__main__":
    main()
