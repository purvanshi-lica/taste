"""
Preprocess multi-dimension ranking & hallucination CSVs for HPS-Contra.

The data folder contains files named ``ui+{a,d}_<Criterion>.csv``:

* **Ranking CSVs** (columns ``eval_round_stage_id, model, rank, prompt_id,
  evaluator, prompt, model_output_image_url``) — pairwise rankings of the
  4 model outputs per prompt, by 5 evaluators, per criterion.

* **Hallucination CSVs** (columns ``evaluator, model, hallucination_value,
  asset_id, prompt_id, hallucination_flag``) — per-asset 3-way ordinal
  rating (0=None, 1=Minor, 2=Major) by 5 evaluators.

We merge the ui+a and ui+d versions of the same criterion under one
*dimension slug* (``preference``, ``typography``, …), so the trainer ends
up with seven scoring heads.  ui+a and ui+d use disjoint images and
prompt_ids; the head sees both as data for the same conceptual task,
distinguished only by the prompt template (``ui+a`` vs ``ui+d``).

For hallucinations we binarise ``hallucination_value > 0`` and aggregate
the soft label across the five evaluators.  Output is a separate
``halluc_train.csv`` / ``halluc_val.csv`` with one row per
(prompt_template, prompt_id, asset).

Outputs (under ``--output-dir``):
    battles_train.csv
    battles_val.csv
    halluc_train.csv
    halluc_val.csv
"""

import argparse
import csv
import random
import re
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import requests
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Filename / dimension parsing
# ---------------------------------------------------------------------------
# ``ui+a_ColorHarmony.csv`` → template=ui+a, criterion=ColorHarmony
_FILENAME_RE = re.compile(r"^ui\+(?P<template>[ad])_(?P<criterion>.+)\.csv$")


def _camel_to_snake(name: str) -> str:
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


# Maps ``criterion`` (as it appears in the filename) → canonical dimension slug
# used as the head id.  Keep this explicit so name drift in future drops
# (e.g. "ColourHarmony" vs "ColorHarmony") doesn't silently create new heads.
DIMENSION_SLUGS: dict[str, str] = {
    "preference": "preference",
    "typography": "typography",
    "ColorHarmony": "color_harmony",
    "MoodAndColorTone": "mood_and_color_tone",
    "VisualHierarchy": "visual_hierarchy",
    "ColorAccuracy": "color_accuracy",
    "SpatialAccuracy": "spatial_accuracy",
}


def parse_filename(path: Path) -> dict | None:
    """Parse a data-folder filename. Returns metadata or ``None`` if unrecognised."""
    m = _FILENAME_RE.match(path.name)
    if not m:
        return None
    template = "ui+" + m.group("template")  # ui+a or ui+d
    criterion = m.group("criterion")

    if criterion == "hallucinations":
        return {"kind": "hallucination", "prompt_template": template}

    if criterion not in DIMENSION_SLUGS:
        # Unknown criterion — treat the lower-cased camel form as the slug
        # so we don't silently drop new dimensions.
        slug = _camel_to_snake(criterion)
        sys.stderr.write(
            f"  Note: unmapped criterion '{criterion}' in {path.name}; "
            f"assigning slug '{slug}'\n"
        )
    else:
        slug = DIMENSION_SLUGS[criterion]

    return {
        "kind": "ranking",
        "prompt_template": template,
        "criterion": criterion,
        "dimension": slug,
    }


# ---------------------------------------------------------------------------
# Image download
# ---------------------------------------------------------------------------
def url_to_filename(url: str) -> str:
    url_id = urlparse(url).path.strip("/").replace("/", "_")
    return f"{url_id}.jpg"


def download_images(urls: Iterable[str], save_dir: Path, max_retries: int = 3) -> dict[str, Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    url_to_path = {u: save_dir / url_to_filename(u) for u in set(urls)}
    print(f"Downloading {len(url_to_path)} unique images into {save_dir} ...")
    failed: list[str] = []

    for url, save_path in tqdm(url_to_path.items(), desc="Downloading images"):
        if save_path.exists():
            continue
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, stream=True, timeout=30)
                resp.raise_for_status()
                with open(save_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"\nFailed to download {url}: {e}")
                    failed.append(url)
                else:
                    time.sleep(2 ** attempt)

    if failed:
        print(f"\n{len(failed)} images failed to download")
    return url_to_path


# ---------------------------------------------------------------------------
# Ranking pipeline
# ---------------------------------------------------------------------------
def load_rankings(csv_path: str) -> dict[tuple, list[dict]]:
    """Group rows by (prompt_id, evaluator, eval_round_stage_id) → 4 entries each."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                row["prompt_id"],
                row["evaluator"],
                row.get("eval_round_stage_id", "0"),
            )
            groups[key].append({
                "model": row["model"],
                "rank": int(row["rank"]),
                "prompt": row["prompt"],
                "image_url": row["model_output_image_url"],
            })

    bad = [k for k, v in groups.items() if len(v) != 4]
    if bad:
        print(
            f"  Warning: {len(bad)} groups have != 4 entries "
            f"(out of {len(groups)} total); they will be skipped"
        )
    return {k: v for k, v in groups.items() if len(v) == 4}


def generate_battles(
    groups: dict[tuple, list[dict]],
    url_to_path: dict[str, Path],
    meta: dict,
    seed: int,
) -> list[dict]:
    """C(4,2)=6 pairwise battles per evaluation group, tagged with dim+template."""
    rng = random.Random(seed)
    battles: list[dict] = []

    for (prompt_id, evaluator, eval_round_id), entries in groups.items():
        prompt = entries[0]["prompt"]

        for a, b in combinations(entries, 2):
            a_path = url_to_path.get(a["image_url"])
            b_path = url_to_path.get(b["image_url"])
            if a_path is None or b_path is None:
                continue

            if rng.random() < 0.5:
                img_a, img_b = a, b
                path_a, path_b = a_path, b_path
            else:
                img_a, img_b = b, a
                path_a, path_b = b_path, a_path

            winner = "A" if img_a["rank"] < img_b["rank"] else "B"

            battles.append({
                "prompt": prompt,
                "image_a": path_a.name,
                "image_b": path_b.name,
                "image_url_a": img_a["image_url"],
                "image_url_b": img_b["image_url"],
                "winner": winner,
                "model_a": img_a["model"],
                "model_b": img_b["model"],
                "rank_a": img_a["rank"],
                "rank_b": img_b["rank"],
                "prompt_id": prompt_id,
                "evaluator": evaluator,
                "eval_round_stage_id": eval_round_id,
                "dimension": meta["dimension"],
                "criterion": meta["criterion"],
                "prompt_template": meta["prompt_template"],
            })

    rng.shuffle(battles)
    return battles


def enrich_battles(battles: list[dict]) -> list[dict]:
    """Aggregate per-evaluator votes into soft labels per (dim, template, prompt, pair)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for b in battles:
        img_lo, img_hi = sorted([b["image_a"], b["image_b"]])
        # Key includes prompt_template so ui+a and ui+d battles for the same
        # criterion never collapse into one row (they use disjoint images
        # and disjoint prompt_ids in practice, but this is the safe key).
        key = (b["dimension"], b["prompt_template"], b["prompt_id"], img_lo, img_hi)
        groups[key].append(b)

    pair_stats: dict[tuple, dict] = {}
    for key, group in groups.items():
        _, _, _, img_lo, _ = key
        votes_lo = 0
        ranks_lo: list[int] = []
        ranks_hi: list[int] = []
        for b in group:
            is_a_lo = b["image_a"] == img_lo
            if is_a_lo:
                ranks_lo.append(int(b["rank_a"]))
                ranks_hi.append(int(b["rank_b"]))
                if b["winner"] == "A":
                    votes_lo += 1
            else:
                ranks_lo.append(int(b["rank_b"]))
                ranks_hi.append(int(b["rank_a"]))
                if b["winner"] == "B":
                    votes_lo += 1
        n = len(group)
        pair_stats[key] = {
            "votes_lo": votes_lo,
            "n_votes": n,
            "agreement": round(max(votes_lo, n - votes_lo) / n, 4),
            "mean_rank_lo": round(sum(ranks_lo) / n, 4),
            "mean_rank_hi": round(sum(ranks_hi) / n, 4),
        }

    enriched: list[dict] = []
    for b in battles:
        img_lo, img_hi = sorted([b["image_a"], b["image_b"]])
        key = (b["dimension"], b["prompt_template"], b["prompt_id"], img_lo, img_hi)
        stats = pair_stats[key]
        is_a_lo = b["image_a"] == img_lo
        if is_a_lo:
            win_rate_a = stats["votes_lo"] / stats["n_votes"]
            mean_rank_a = stats["mean_rank_lo"]
            mean_rank_b = stats["mean_rank_hi"]
        else:
            win_rate_a = 1.0 - stats["votes_lo"] / stats["n_votes"]
            mean_rank_a = stats["mean_rank_hi"]
            mean_rank_b = stats["mean_rank_lo"]
        enriched.append({
            **b,
            "win_rate_a": round(win_rate_a, 4),
            "n_votes": stats["n_votes"],
            "agreement": stats["agreement"],
            "mean_rank_a": mean_rank_a,
            "mean_rank_b": mean_rank_b,
            "agreement_bucket": agreement_bucket(stats["agreement"]),
        })
    return enriched


def agreement_bucket(agreement: float) -> str:
    """Bucket the cross-evaluator agreement ratio.

    With 5 evaluators on a binary winner choice, agreement ∈ {0.6, 0.8, 1.0}:

    * ``unanimous`` — 5-0  (≈ 1.0)
    * ``majority``  — 4-1  (≈ 0.8)
    * ``split``     — 3-2  (≈ 0.6) — genuinely ambiguous

    Boundaries at 0.95 / 0.7 keep the three classes distinct on the
    discrete grid produced by 5 evaluators.
    """
    if agreement >= 0.95:
        return "unanimous"
    if agreement >= 0.7:
        return "majority"
    return "split"


def dedupe_battles(battles: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    out: list[dict] = []
    for b in battles:
        img_lo, img_hi = sorted([b["image_a"], b["image_b"]])
        key = (b["dimension"], b["prompt_template"], b["prompt_id"], img_lo, img_hi)
        if key in seen:
            continue
        seen.add(key)
        row = dict(b)
        row.pop("evaluator", None)
        row.pop("eval_round_stage_id", None)
        out.append(row)
    return out


def stratified_split_by_prompt(
    battles: list[dict],
    val_split: float,
    seed: int,
    group_keys: tuple[str, ...] = ("dimension", "prompt_template"),
) -> tuple[list[dict], list[dict]]:
    """Split prompts per (dimension, prompt_template), preferring val prompts
    that span all agreement buckets.  Train/val are disjoint at prompt_id."""
    rng = random.Random(seed)

    by_group: dict[tuple, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for b in battles:
        gkey = tuple(b[k] for k in group_keys)
        by_group[gkey][b["prompt_id"]].add(b["agreement_bucket"])

    val_keys: set[tuple] = set()
    for gkey, pid_buckets in by_group.items():
        prompts = list(pid_buckets.keys())
        rng.shuffle(prompts)
        n_total = len(prompts)
        n_val = max(1, int(round(n_total * val_split)))

        chosen: list[str] = []
        covered: set[str] = set()
        remaining = list(prompts)

        for buck in ("unanimous", "majority", "split"):
            for pid in list(remaining):
                if buck in pid_buckets[pid] and buck not in covered:
                    chosen.append(pid)
                    covered.update(pid_buckets[pid])
                    remaining.remove(pid)
                    break

        for pid in remaining:
            if len(chosen) >= n_val:
                break
            chosen.append(pid)

        for pid in chosen:
            val_keys.add(gkey + (pid,))

    train, val = [], []
    for b in battles:
        k = tuple(b[g] for g in group_keys) + (b["prompt_id"],)
        (val if k in val_keys else train).append(b)
    return train, val


# ---------------------------------------------------------------------------
# Hallucination pipeline
# ---------------------------------------------------------------------------
def load_hallucinations(csv_path: str) -> dict[tuple, list[dict]]:
    """Group hallucination rows per asset; each group has 5 evaluator rows.

    The CSVs here don't carry the prompt text or the image URL — they only
    have ``asset_id`` and ``prompt_id``.  We resolve the prompt text and the
    image URL by joining against the matching ranking CSV (same prompt_template
    family) downstream in :func:`build_hallucination_rows`.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    with open(csv_path, "r") as f:
        for r in csv.DictReader(f):
            key = (r["prompt_id"], r["model"], r["asset_id"])
            groups[key].append({
                "evaluator": r["evaluator"],
                "value": int(r["hallucination_value"]),
                "flag": r["hallucination_flag"],
            })
    return groups


def build_hallucination_rows(
    halluc_groups: dict[tuple, list[dict]],
    asset_lookup: dict[tuple, dict],
    template: str,
) -> list[dict]:
    """Produce one row per asset with soft binary label ``halluc_rate``.

    ``asset_lookup`` is a ``(prompt_id, model) → {prompt, image_url}`` map
    built from the matching ranking CSVs (same template family) so we can
    attach the prompt text and image URL to each hallucination row.
    """
    rows: list[dict] = []
    missing_join = 0
    for (prompt_id, model, asset_id), votes in halluc_groups.items():
        n = len(votes)
        binary = [1 if v["value"] > 0 else 0 for v in votes]
        mean_value = sum(v["value"] for v in votes) / n
        halluc_rate = sum(binary) / n

        meta = asset_lookup.get((prompt_id, model))
        if meta is None:
            missing_join += 1
            continue

        rows.append({
            "asset_id": asset_id,
            "prompt_id": prompt_id,
            "model": model,
            "prompt": meta["prompt"],
            "image_url": meta["image_url"],
            "image": Path(url_to_filename(meta["image_url"])).name,
            "n_evaluators": n,
            "mean_value": round(mean_value, 4),
            "halluc_rate": round(halluc_rate, 4),
            "label": int(round(halluc_rate)),  # majority vote
            "agreement": round(max(halluc_rate, 1 - halluc_rate), 4),
            "agreement_bucket": agreement_bucket(max(halluc_rate, 1 - halluc_rate)),
            "prompt_template": template,
        })
    if missing_join:
        print(
            f"  Note: {missing_join} hallucination rows could not be joined "
            f"to a (prompt_id, model) in the matching ranking CSVs"
        )
    return rows


def stratified_split_hallucinations(
    rows: list[dict],
    val_split: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Split halluc rows by prompt_id within each prompt_template, stratifying on bucket."""
    return stratified_split_by_prompt(
        rows, val_split=val_split, seed=seed, group_keys=("prompt_template",)
    )


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
_BATTLE_FIELD_ORDER = [
    "dimension", "criterion", "prompt_template",
    "prompt", "image_a", "image_b", "image_url_a", "image_url_b",
    "winner", "model_a", "model_b", "rank_a", "rank_b",
    "prompt_id", "evaluator", "eval_round_stage_id",
    "win_rate_a", "n_votes", "agreement", "agreement_bucket",
    "mean_rank_a", "mean_rank_b",
]

_HALLUC_FIELD_ORDER = [
    "prompt_template", "prompt_id", "model", "asset_id",
    "prompt", "image", "image_url",
    "n_evaluators", "mean_value", "halluc_rate", "label",
    "agreement", "agreement_bucket",
]


def _save(rows: list[dict], path: Path, field_order: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    present = set(rows[0].keys())
    fields = [f for f in field_order if f in present]
    fields += sorted(present - set(fields))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Preprocess multi-dimension ranking + hallucination CSVs"
    )
    parser.add_argument("--input-dir", type=str, default="data",
                        help="Directory of ui+a_*.csv / ui+d_*.csv files")
    parser.add_argument("--input-csv", type=str, nargs="*", default=None,
                        help="Explicit list of CSVs (overrides --input-dir)")
    parser.add_argument("--image-dir", type=str, default="data/images")
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument(
        "--dedupe", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "Keep one row per unique pair after evaluator aggregation. "
            "Default is False: emit one row per (evaluator, pair). The "
            "aggregate stats (win_rate_a, agreement, agreement_bucket) are "
            "still attached to every row, so the trainer can choose to "
            "weight by agreement and the validator can dedupe at read-time "
            "for clean per-pair metrics."
        ),
    )
    parser.add_argument(
        "--dedupe-val", action="store_true",
        help=(
            "Even when --no-dedupe is set, write the val split as one row "
            "per pair (the train split keeps all 5 evaluator rows). Useful "
            "when you want clean per-pair val metrics."
        ),
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── Resolve inputs ────────────────────────────────────────────────
    if args.input_csv:
        input_paths = [Path(p).resolve() for p in args.input_csv]
    else:
        d = Path(args.input_dir).resolve()
        input_paths = sorted(d.glob("*.csv"))
    if not input_paths:
        parser.error("No input CSVs found.")

    image_dir = Path(args.image_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    # ── Classify files ─────────────────────────────────────────────────
    ranking_files: list[tuple[Path, dict]] = []
    halluc_files: list[tuple[Path, dict]] = []
    for p in input_paths:
        meta = parse_filename(p)
        if meta is None:
            print(f"  Skipping unrecognised file: {p.name}")
            continue
        if meta["kind"] == "ranking":
            ranking_files.append((p, meta))
        else:
            halluc_files.append((p, meta))

    if not ranking_files and not halluc_files:
        parser.error("No recognised CSVs.")

    print(f"Found {len(ranking_files)} ranking CSV(s) and {len(halluc_files)} hallucination CSV(s):")
    for p, m in ranking_files:
        print(f"  · ranking      {m['prompt_template']:<5}  {m['criterion']:<20} → dim '{m['dimension']}'")
    for p, m in halluc_files:
        print(f"  · hallucination {m['prompt_template']:<5}  {p.name}")

    # ── Load rankings; collect URLs ───────────────────────────────────
    per_file_groups: list[tuple[dict, dict[tuple, list[dict]]]] = []
    asset_lookup: dict[str, dict[tuple, dict]] = defaultdict(dict)  # template → (pid, model) → {prompt, url}
    all_urls: set[str] = set()

    for path, meta in ranking_files:
        groups = load_rankings(str(path))
        per_file_groups.append((meta, groups))
        for (pid, _ev, _round), entries in groups.items():
            for e in entries:
                all_urls.add(e["image_url"])
                # Hallucination CSVs only carry (prompt_id, model, asset_id);
                # they share asset_ids with this template's ranking CSVs, so
                # we build a per-template (prompt_id, model) → {prompt, url}
                # map to attach the prompt text + image url downstream.
                asset_lookup[meta["prompt_template"]][(pid, e["model"])] = {
                    "prompt": e["prompt"],
                    "image_url": e["image_url"],
                }

    # ── Download all images ────────────────────────────────────────────
    if args.skip_download:
        url_to_path = {u: image_dir / url_to_filename(u) for u in all_urls}
        missing = [u for u, p in url_to_path.items() if not p.exists()]
        if missing:
            print(
                f"  Warning: --skip-download but {len(missing)}/{len(url_to_path)} "
                f"images are not present in {image_dir}",
                file=sys.stderr,
            )
    else:
        url_to_path = download_images(all_urls, image_dir)

    # ── Build battles, aggregate, dedupe, per dimension+template ──────
    print("\nBuilding battles ...")
    battles: list[dict] = []
    for meta, groups in per_file_groups:
        b = generate_battles(groups, url_to_path, meta, seed=args.seed)
        b = enrich_battles(b)
        if args.dedupe:
            b = dedupe_battles(b)
        bk = defaultdict(int)
        for x in b:
            bk[x["agreement_bucket"]] += 1
        print(
            f"  → {meta['dimension']:<22} ({meta['prompt_template']:<5}): {len(b):>4} pairs  "
            f"(unanimous={bk['unanimous']:>3}, majority={bk['majority']:>3}, split={bk['split']:>3})"
        )
        battles.extend(b)

    train_battles, val_battles = stratified_split_by_prompt(
        battles, val_split=args.val_split, seed=args.seed,
        group_keys=("dimension", "prompt_template"),
    )
    if args.dedupe_val and not args.dedupe:
        val_battles = dedupe_battles(val_battles)
    print(f"\nBattles: {len(train_battles)} train + {len(val_battles)} val")

    print("Val composition (per dimension × prompt_template):")
    bucket_count: dict[tuple[str, str, str], int] = defaultdict(int)
    for b in val_battles:
        bucket_count[(b["dimension"], b["prompt_template"], b["agreement_bucket"])] += 1
    for (dim, tmpl), _ in sorted({(d, t): None for (d, t, _) in bucket_count}.items()):
        u = bucket_count[(dim, tmpl, "unanimous")]
        m = bucket_count[(dim, tmpl, "majority")]
        s = bucket_count[(dim, tmpl, "split")]
        print(f"  {dim:<22} {tmpl:<5}  unanimous={u:>3}  majority={m:>3}  split={s:>3}")

    _save(train_battles, output_dir / "battles_train.csv", _BATTLE_FIELD_ORDER)
    _save(val_battles, output_dir / "battles_val.csv", _BATTLE_FIELD_ORDER)

    # ── Hallucinations ─────────────────────────────────────────────────
    halluc_rows: list[dict] = []
    if halluc_files:
        print("\nBuilding hallucination rows ...")
        for path, meta in halluc_files:
            groups = load_hallucinations(str(path))
            template = meta["prompt_template"]
            rows = build_hallucination_rows(groups, asset_lookup[template], template)
            print(
                f"  → {path.name}: {len(rows)} assets  "
                f"(positives={sum(r['label'] for r in rows)}/{len(rows)})"
            )
            halluc_rows.extend(rows)

        halluc_train, halluc_val = stratified_split_hallucinations(
            halluc_rows, val_split=args.val_split, seed=args.seed
        )
        print(f"\nHallucinations: {len(halluc_train)} train + {len(halluc_val)} val")

        _save(halluc_train, output_dir / "halluc_train.csv", _HALLUC_FIELD_ORDER)
        _save(halluc_val, output_dir / "halluc_val.csv", _HALLUC_FIELD_ORDER)

    print(f"\nSaved outputs under {output_dir}")


if __name__ == "__main__":
    main()
