"""Command-line entry point for taste-scorer.

After ``pip install -e .``, the ``taste-score`` console script is
available::

    # Score a CSV of (prompt, image_a, image_b) pairs.
    taste-score score INPUT.csv -o OUTPUT.csv \\
        --checkpoint path/to/best_pairwise/best \\
        --image-dir path/to/images/

    # Aggregate a scored CSV into a per-model × per-dimension leaderboard.
    # Requires model_a / model_b on the scored CSV (declared as optional
    # input columns and passed straight through by ``score``).
    taste-score leaderboard SCORED.csv -o LEADERBOARD.csv

    # ...or do both in one shot:
    taste-score score INPUT.csv -o OUTPUT.csv \\
        --checkpoint path/to/best_pairwise/best \\
        --image-dir path/to/images/ \\
        --leaderboard LEADERBOARD.csv

For ad-hoc use without installing, ``scripts/score.py`` does the same
thing without requiring entry-point installation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from taste_scorer.scorer import (
    PreferenceScorer,
    compute_leaderboard,
    format_leaderboard,
)


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="taste-score",
        description=(
            "Score image pairs along multiple ranking dimensions with a "
            "trained TASTE preference model, and (optionally) aggregate "
            "the scored pairs into a per-model leaderboard."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser(
        "score",
        help="Score a CSV of (prompt, image_a, image_b) rows.",
        description=(
            "Read an input CSV with columns 'prompt', 'image_a', "
            "'image_b' (and optionally 'pair_id', 'model_a', 'model_b') "
            "and write an output CSV that adds 'prob_a_wins_<dim>' for "
            "each model dimension, plus 'halluc_prob_a' / "
            "'halluc_prob_b' if the checkpoint has a hallucination head. "
            "If 'model_a' and 'model_b' are present and --leaderboard is "
            "set, also write a per-model leaderboard."
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
    s.add_argument(
        "--leaderboard", default=None,
        help=(
            "Optional path to also write a per-model × per-dimension "
            "leaderboard CSV.  Requires 'model_a' / 'model_b' columns on "
            "the input CSV; ignored otherwise (with a warning)."
        ),
    )

    lb = sub.add_parser(
        "leaderboard",
        help="Aggregate a scored CSV into a per-model × per-dimension table.",
        description=(
            "Read a scored CSV (output of `taste-score score`) that "
            "carries 'model_a' / 'model_b' columns and write a "
            "leaderboard CSV: one row per generator, one column per "
            "ranking dimension, plus 'overall', 'n_pairs', and (when "
            "available) 'halluc_rate'.  Models are sorted by 'overall' "
            "descending."
        ),
    )
    lb.add_argument("scored_csv", help="Path to the scored CSV.")
    lb.add_argument(
        "-o", "--output-csv", required=True,
        help="Path to write the leaderboard CSV.",
    )
    lb.add_argument(
        "--no-halluc", action="store_true",
        help=(
            "Skip the per-model halluc_rate column even if the scored "
            "CSV carries hallucination probabilities."
        ),
    )
    lb.add_argument(
        "--quiet", action="store_true",
        help="Suppress the formatted leaderboard printout.",
    )

    return p


def _run_leaderboard(
    scored: pd.DataFrame,
    output_path: Path,
    include_halluc: bool,
    quiet: bool,
) -> None:
    table = compute_leaderboard(scored, include_halluc=include_halluc)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path)
    if not quiet:
        print()
        print("Per-model leaderboard:")
        print(format_leaderboard(table))
    print(f"Wrote leaderboard to {output_path}")


def main(argv: list[str] | None = None) -> int:
    args = _make_parser().parse_args(argv)

    if args.command == "score":
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
        print(f"Hallucination head: {'yes' if scorer.has_halluc_head else 'no'}")

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

        if args.leaderboard:
            lb_path = Path(args.leaderboard)
            if "model_a" in df_out.columns and "model_b" in df_out.columns:
                _run_leaderboard(
                    df_out, lb_path, include_halluc=True, quiet=False,
                )
            else:
                print(
                    "WARNING: --leaderboard requested but input CSV is "
                    "missing 'model_a' / 'model_b' columns; skipping.",
                    file=sys.stderr,
                )
        return 0

    if args.command == "leaderboard":
        scored_path = Path(args.scored_csv)
        if not scored_path.is_file():
            print(f"Scored CSV not found: {scored_path}", file=sys.stderr)
            return 1

        scored = pd.read_csv(scored_path)
        _run_leaderboard(
            scored,
            Path(args.output_csv),
            include_halluc=not args.no_halluc,
            quiet=args.quiet,
        )
        return 0

    print(f"Unknown command: {args.command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
