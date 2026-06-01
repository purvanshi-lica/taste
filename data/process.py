"""Processing utilities for the TASTE dataset on the Hugging Face Hub.

Source dataset: https://huggingface.co/datasets/purvanshi/TASTE

The dataset is published as a set of canonical parquet tables (plus the raw
images and a couple of pre-joined browseable views):

    prompts.parquet         prompt_id, track, dimension, prompt_id_src, prompt_text
    assets.parquet          asset_id, model, image_url, image_path, track
    rankings.parquet        eval_round_stage_id, dimension, track, prompt_id,
                            asset_id, evaluator_id, rank
    hallucinations.parquet  track, prompt_id_src, asset_id, evaluator_id,
                            hallucination_value, hallucination_flag
    evaluators.parquet      evaluator_id, tracks, n_ranking_rows, n_halluc_rows

This module turns those canonical tables into the two derived artifacts the
rest of the repo consumes:

  * per-(track, dimension) long-format ranking CSVs  ->  ``analysis/``
  * pairwise Bradley-Terry battles                   ->  ``taste-scorer/``

Subcommands
-----------
    download   Snapshot the dataset (parquet tables + images) into a local dir.
    rankings   Write one ranking CSV per (track, dimension).
    battles    Derive pairwise battles (winner-vs-loser) from the rankings.

Run ``python process.py <subcommand> --help`` for the per-command flags.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HF_REPO_ID = "purvanshi/TASTE"

HERE = Path(__file__).resolve().parent
DEFAULT_RAW_DIR = HERE / "raw"

# Canonical parquet tables we rely on (filename -> friendly key).
CANONICAL_TABLES = {
    "prompts": "prompts.parquet",
    "assets": "assets.parquet",
    "rankings": "rankings.parquet",
    "hallucinations": "hallucinations.parquet",
    "evaluators": "evaluators.parquet",
}

# (track, dimension) pairs that exist in the corpus, mapped to the slug the
# analysis framework uses. preference + typography are evaluated under both
# tracks; the remaining dimensions are exclusive to one track.
DIMENSION_SLUGS = {
    ("aesthetics", "preference"): "aesthetics_preference",
    ("aesthetics", "mood_and_color_tone"): "aesthetics_mood",
    ("aesthetics", "visual_hierarchy"): "aesthetics_visual_hier",
    ("aesthetics", "color_harmony"): "aesthetics_color_harmony",
    ("aesthetics", "typography"): "aesthetics_typography",
    ("descriptions", "preference"): "descriptions_preference",
    ("descriptions", "color_accuracy"): "descriptions_color_acc",
    ("descriptions", "spatial_accuracy"): "descriptions_spatial_acc",
    ("descriptions", "typography"): "descriptions_typography",
}

# Columns the analysis ranking CSVs expect (see ../analysis/README.md).
RANKING_CSV_COLUMNS = [
    "eval_round_stage_id",
    "model",
    "rank",
    "prompt_id",
    "evaluator",
    "prompt",
    "model_output_image_url",
]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _resolve_table(raw_dir: Path, key: str) -> str:
    """Return a path/URI for a canonical table.

    Prefers a local snapshot under ``raw_dir``; otherwise falls back to reading
    directly from the Hub via the ``hf://`` filesystem (requires
    ``huggingface_hub``).
    """
    filename = CANONICAL_TABLES[key]
    local = raw_dir / filename
    if local.exists():
        return str(local)
    # Fall back to streaming straight from the Hub.
    return f"hf://datasets/{HF_REPO_ID}/{filename}"


def load_tables(raw_dir: Path, keys=("prompts", "assets", "rankings")):
    """Load the requested canonical tables as a dict of DataFrames."""
    tables = {}
    for key in keys:
        src = _resolve_table(raw_dir, key)
        try:
            tables[key] = pd.read_parquet(src)
        except Exception as exc:  # noqa: BLE001 - surface a useful hint
            raise SystemExit(
                f"Failed to read '{key}' table from {src!r}: {exc}\n"
                "Run `python process.py download` first, or `pip install "
                "huggingface_hub` to stream directly from the Hub."
            ) from exc
    return tables


def _join_rankings(tables) -> pd.DataFrame:
    """Join rankings with prompt text and asset (model + image url)."""
    rankings = tables["rankings"]
    prompts = tables["prompts"][["prompt_id", "prompt_text"]]
    assets = tables["assets"][["asset_id", "model", "image_url"]]
    df = rankings.merge(prompts, on="prompt_id", how="left")
    df = df.merge(assets, on="asset_id", how="left")
    return df


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
def cmd_download(args) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub is required for `download`.\n"
            "  pip install -r requirements.txt"
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    allow_patterns = None if args.with_images else ["*.parquet", "*.md", "LICENSE*"]
    print(f"Downloading {HF_REPO_ID} -> {out}")
    if not args.with_images:
        print("  (parquet tables only; pass --with-images to also fetch images/)")
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=str(out),
        allow_patterns=allow_patterns,
    )
    print("Done.")


def cmd_rankings(args) -> None:
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = load_tables(raw_dir, keys=("prompts", "assets", "rankings"))
    df = _join_rankings(tables)

    written = []
    for (track, dimension), slug in DIMENSION_SLUGS.items():
        sub = df[(df["track"] == track) & (df["dimension"] == dimension)].copy()
        if sub.empty:
            print(f"  [skip] {slug}: no rows for ({track}, {dimension})")
            continue
        sub = sub.rename(
            columns={
                "evaluator_id": "evaluator",
                "prompt_text": "prompt",
                "image_url": "model_output_image_url",
            }
        )
        sub = sub[RANKING_CSV_COLUMNS]
        path = out_dir / f"{slug}.csv"
        sub.to_csv(path, index=False)
        written.append((slug, len(sub), sub["prompt_id"].nunique()))
        print(f"  {slug:28s} rows={len(sub):5d} prompts={sub['prompt_id'].nunique():4d}")

    if not written:
        raise SystemExit("No ranking CSVs written — check the input tables.")
    print(f"\nWrote {len(written)} ranking CSV(s) to {out_dir}/")


def cmd_battles(args) -> None:
    raw_dir = Path(args.raw_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tables = load_tables(raw_dir, keys=("prompts", "assets", "rankings"))
    rankings = tables["rankings"]

    # Self-join within each (session, evaluator, prompt, dimension, track) group
    # and keep ordered pairs where rank_a < rank_b: A is the preferred design.
    keys = ["eval_round_stage_id", "evaluator_id", "prompt_id", "dimension", "track"]
    pairs = rankings.merge(rankings, on=keys, suffixes=("_a", "_b"))
    pairs = pairs[pairs["rank_a"] < pairs["rank_b"]].copy()

    assets = tables["assets"][["asset_id", "model", "image_url"]]
    prompts = tables["prompts"][["prompt_id", "prompt_text"]]
    pairs = pairs.merge(
        assets.rename(columns={"asset_id": "asset_id_a", "model": "model_a",
                               "image_url": "image_a"}),
        on="asset_id_a", how="left",
    )
    pairs = pairs.merge(
        assets.rename(columns={"asset_id": "asset_id_b", "model": "model_b",
                               "image_url": "image_b"}),
        on="asset_id_b", how="left",
    )
    pairs = pairs.merge(prompts, on="prompt_id", how="left")

    pairs = pairs.rename(columns={"prompt_text": "prompt"})
    pairs.insert(0, "pair_id", range(1, len(pairs) + 1))
    pairs["winner"] = "a"  # image_a is preferred by construction

    out_cols = [
        "pair_id", "track", "dimension", "prompt", "prompt_id",
        "image_a", "image_b", "model_a", "model_b",
        "asset_id_a", "asset_id_b", "evaluator_id", "rank_a", "rank_b", "winner",
    ]
    pairs = pairs[out_cols]
    pairs.to_csv(out_path, index=False)
    print(f"Wrote {len(pairs)} battles to {out_path}")
    print(f"  tracks: {sorted(pairs['track'].unique())}")
    print(f"  dimensions: {sorted(pairs['dimension'].unique())}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="process.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download", help="snapshot the HF dataset locally")
    p_dl.add_argument("--out", default=str(DEFAULT_RAW_DIR),
                      help=f"output dir (default: {DEFAULT_RAW_DIR})")
    p_dl.add_argument("--with-images", action="store_true",
                      help="also download the images/ folder (~1.6 GB)")
    p_dl.set_defaults(func=cmd_download)

    p_rank = sub.add_parser("rankings", help="write per-(track, dimension) ranking CSVs")
    p_rank.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR),
                        help=f"local snapshot dir (default: {DEFAULT_RAW_DIR}); "
                             "if missing, streams from the Hub")
    p_rank.add_argument("--out", default=str(HERE / "rankings"),
                        help="output dir for the CSVs")
    p_rank.set_defaults(func=cmd_rankings)

    p_bat = sub.add_parser("battles", help="derive pairwise battles from rankings")
    p_bat.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR),
                       help=f"local snapshot dir (default: {DEFAULT_RAW_DIR}); "
                            "if missing, streams from the Hub")
    p_bat.add_argument("--out", default=str(HERE / "battles.csv"),
                       help="output CSV path")
    p_bat.set_defaults(func=cmd_battles)

    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
