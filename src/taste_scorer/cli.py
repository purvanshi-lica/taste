"""Command-line entry point for taste-scorer.

After ``pip install -e .``, the ``taste-score`` console script is
available::

    taste-score score INPUT.csv -o OUTPUT.csv \\
        --checkpoint path/to/best_pairwise/best \\
        --image-dir path/to/images/

For ad-hoc use without installing, ``scripts/score.py`` does the same
thing without requiring entry-point installation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from taste_scorer.scorer import PreferenceScorer


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="taste-score",
        description=(
            "Score image pairs along multiple ranking dimensions with a "
            "trained TASTE preference model."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser(
        "score",
        help="Score a CSV of (prompt, image_a, image_b) rows.",
        description=(
            "Read an input CSV with columns 'prompt', 'image_a', "
            "'image_b' (and optionally 'pair_id'), and write an output "
            "CSV that adds 'prob_a_wins_<dim>' for each model dimension, "
            "plus 'halluc_prob_a' / 'halluc_prob_b' if the checkpoint "
            "has a hallucination head."
        ),
    )
    s.add_argument("input_csv", help="Path to input CSV.")
    s.add_argument(
        "-o", "--output-csv", required=True,
        help="Path to write the scored CSV.",
    )
    s.add_argument(
        "--checkpoint", required=True,
        help=(
            "Path to a checkpoint directory containing heads.pt, "
            "halluc_head.pt (optional), meta.json, and optionally "
            "lora_adapter/."
        ),
    )
    s.add_argument(
        "--image-dir", default=None,
        help=(
            "Root directory for relative image paths in the input CSV.  "
            "Absolute paths in the CSV bypass this."
        ),
    )
    s.add_argument(
        "--device", default=None,
        help="Torch device ('cuda', 'cpu', 'mps').  Auto-detected if omitted.",
    )
    s.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for the head forward pass (default: 64).",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    args = _make_parser().parse_args(argv)
    if args.command != "score":
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 2

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    if not input_path.is_file():
        print(f"Input CSV not found: {input_path}", file=sys.stderr)
        return 1

    print(f"Loading checkpoint from {args.checkpoint} ...")
    scorer = PreferenceScorer.from_checkpoint(args.checkpoint, device=args.device)
    print(
        f"Loaded {len(scorer.dimensions)} dimension head(s): "
        f"{scorer.dimensions}"
    )
    if scorer.has_halluc_head:
        print("Hallucination head: yes")
    else:
        print("Hallucination head: no")

    print(f"Reading input from {input_path}")
    df = pd.read_csv(input_path)
    print(f"  {len(df)} rows")

    df_out = scorer.score_pairs(
        df,
        image_dir=args.image_dir,
        batch_size=args.batch_size,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"Wrote scored CSV to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
