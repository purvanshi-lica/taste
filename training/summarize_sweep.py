"""Compare the runs produced by ``sweep_overfit.sh``.

Reads each ``<sweep-root>/<key>/history.json`` produced by ``train.py``,
picks the best epoch per run by ``val_acc``, and prints a single comparison
table plus a recommendation block.

Usage::

    python summarize_sweep.py
    python summarize_sweep.py --sweep-root checkpoints/sweep
    python summarize_sweep.py --rank-by kendall   # default: val_acc
    python summarize_sweep.py --json out.json     # also write a machine-readable summary

A run's "score" is its val_acc at the epoch with highest val_acc.  The
recommendation block flags overfitting (train_acc - val_acc gap > 0.15)
and tells you which config to iterate on next.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_history(run_dir: Path) -> list[dict] | None:
    path = run_dir / "history.json"
    if not path.exists():
        return None
    with open(path) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return None


def _best_epoch_by_val_acc(history: list[dict]) -> dict:
    return max(history, key=lambda r: r.get("val_acc", -1.0))


def _per_dim_mean(per_dim: dict[str, dict], key: str) -> float:
    if not per_dim:
        return float("nan")
    vals = [m.get(key, float("nan")) for m in per_dim.values()]
    vals = [v for v in vals if v == v]  # drop NaN
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def _bucket_acc(overall: dict, bucket: str) -> tuple[float, int]:
    accs = overall.get("accuracy_by_agreement", {})
    if bucket not in accs:
        return float("nan"), 0
    val = accs[bucket]
    # Stored as [acc, count] in JSON (tuple → list).
    if isinstance(val, (list, tuple)) and len(val) == 2:
        return float(val[0]), int(val[1])
    return float("nan"), 0


def _summarize_run(key: str, run_dir: Path) -> dict[str, Any] | None:
    history = _load_history(run_dir)
    if not history:
        return None
    best = _best_epoch_by_val_acc(history)
    overall = best.get("overall", {})
    per_dim = best.get("per_dim", {})
    halluc = best.get("halluc")  # dict-keyed-by-template or None

    halluc_auc = float("nan")
    if isinstance(halluc, dict) and halluc:
        aucs = [m.get("auc", float("nan")) for m in halluc.values()]
        aucs = [a for a in aucs if a == a]
        if aucs:
            halluc_auc = sum(aucs) / len(aucs)

    unanimous_acc, unanimous_n = _bucket_acc(overall, "unanimous")
    majority_acc, majority_n = _bucket_acc(overall, "majority")
    split_acc, split_n = _bucket_acc(overall, "split")

    return {
        "key": key,
        "best_epoch": int(best.get("epoch", -1)),
        "n_epochs": len(history),
        "train_acc": float(best.get("train_acc", float("nan"))),
        "val_acc": float(best.get("val_acc", float("nan"))),
        "train_val_gap": (
            float(best.get("train_acc", float("nan")))
            - float(best.get("val_acc", float("nan")))
        ),
        "train_bt_loss": float(best.get("train_bt_loss", float("nan"))),
        "val_bt_loss": float(best.get("val_bt_loss", float("nan"))),
        "mean_kendall_tau": float(overall.get("mean_kendall_tau", float("nan"))),
        "kendall_per_dim": _per_dim_mean(per_dim, "mean_kendall_tau"),
        "unanimous_acc": unanimous_acc,
        "unanimous_n": unanimous_n,
        "majority_acc": majority_acc,
        "majority_n": majority_n,
        "split_acc": split_acc,
        "split_n": split_n,
        "halluc_auc": halluc_auc,
    }


def _fmt(v: float, width: int = 6, prec: int = 3) -> str:
    if v != v:  # NaN
        return "  -  ".center(width)
    return f"{v:>{width}.{prec}f}"


def _print_table(rows: list[dict[str, Any]], rank_by: str) -> None:
    rows = sorted(rows, key=lambda r: -r.get(rank_by, -1.0))
    header = (
        f"{'run':<7s}  "
        f"{'epoch':>5s}  "
        f"{'train_acc':>9s}  "
        f"{'val_acc':>7s}  "
        f"{'gap':>6s}  "
        f"{'train_loss':>10s}  "
        f"{'val_loss':>9s}  "
        f"{'tau':>6s}  "
        f"{'unanim':>6s}  "
        f"{'major':>6s}  "
        f"{'split':>6s}  "
        f"{'halluc_auc':>10s}"
    )
    print(header)
    print("─" * len(header))
    for i, r in enumerate(rows):
        marker = "*" if i == 0 else " "
        print(
            f"{marker}{r['key']:<6s}  "
            f"{r['best_epoch']:>5d}  "
            f"{_fmt(r['train_acc'], 9)}  "
            f"{_fmt(r['val_acc'], 7)}  "
            f"{_fmt(r['train_val_gap'], 6)}  "
            f"{_fmt(r['train_bt_loss'], 10, 4)}  "
            f"{_fmt(r['val_bt_loss'], 9, 4)}  "
            f"{_fmt(r['mean_kendall_tau'], 6)}  "
            f"{_fmt(r['unanimous_acc'], 6)}  "
            f"{_fmt(r['majority_acc'], 6)}  "
            f"{_fmt(r['split_acc'], 6)}  "
            f"{_fmt(r['halluc_auc'], 10)}"
        )
    print()


def _recommend(rows: list[dict[str, Any]], rank_by: str) -> None:
    if not rows:
        return
    ranked = sorted(rows, key=lambda r: -r.get(rank_by, -1.0))
    winner = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    by_key = {r["key"]: r for r in rows}
    A = by_key.get("A")

    print("── Recommendation ─────────────────────────────────────────────")
    print(f"Best by {rank_by}: {winner['key']}  "
          f"(val_acc={winner['val_acc']:.3f}, "
          f"tau={winner['mean_kendall_tau']:.3f}, "
          f"gap={winner['train_val_gap']:+.3f})")
    if runner_up:
        print(f"Runner-up:     {runner_up['key']}  "
              f"(val_acc={runner_up['val_acc']:.3f}, "
              f"tau={runner_up['mean_kendall_tau']:.3f}, "
              f"gap={runner_up['train_val_gap']:+.3f})")

    # Sanity / overfit checks ------------------------------------------------
    notes: list[str] = []

    if winner["val_acc"] < 0.55:
        notes.append(
            "Even the best run is barely above chance (val_acc<0.55). "
            "Suspect a data / signal problem before tuning further: "
            "rerun check_embeddings.py on the train CSV, and inspect a few "
            "per-evaluator labels for sanity."
        )
    elif winner["val_acc"] < 0.60:
        notes.append(
            "Best val_acc < 0.60.  Modest signal but probably leaving "
            "performance on the table.  Try Run D (LoRA) if you haven't, "
            "or a longer schedule (MLP_EPOCHS=2000)."
        )
    if winner["mean_kendall_tau"] < 0.05:
        notes.append(
            "Kendall τ near 0: the model is barely correlated with human "
            "rankings even on the best run.  Same prescription as above — "
            "look at signal, not capacity."
        )
    if winner["train_val_gap"] > 0.15:
        notes.append(
            "Train-val gap > 0.15 → still overfitting.  Try a smaller / "
            "more regularised variant of the winner (lower hidden_dim, "
            "higher dropout, higher weight_decay)."
        )
    elif winner["train_val_gap"] < 0.02 and winner["val_acc"] > 0.55:
        notes.append(
            "Train-val gap < 0.02 → you're under-fitting; the heads have "
            "room to grow.  Try a slightly *larger* variant of the winner "
            "(more capacity, lower regularisation)."
        )

    # Sanity ablations -------------------------------------------------------
    E = by_key.get("E")
    if A and E and E["val_acc"] > A["val_acc"] + 0.005:
        notes.append(
            "Run E (soft labels) outperforms Run A (hard labels).  Our "
            "default of per-row hard labels is wrong for this data; rerun "
            "your winner with --soft-labels."
        )
    F = by_key.get("F")
    if A and F and F["val_acc"] > A["val_acc"] - 0.005:
        notes.append(
            "Run F (no agreement weighting) is not worse than Run A.  "
            "Agreement weighting isn't helping; simplify by passing "
            "--no-agreement-weighting on future runs."
        )

    if notes:
        print()
        for n in notes:
            print(f"  • {n}")
    print()
    print("Next steps:")
    print(f"  1. Iterate from {winner['key']}: take its config and perturb")
    print("     one knob at a time (capacity, dropout, weight-decay, lr).")
    print("  2. Generate the eval HTML report on the winner's checkpoint:")
    print(f"       python inference.py eval \\\n"
          f"           --checkpoint <sweep-root>/{winner['key']}/best \\\n"
          f"           --val-csv <your val CSV> \\\n"
          f"           --halluc-val-csv <your halluc val CSV> \\\n"
          f"           --image-dir <your image dir>")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-root", default="checkpoints/sweep",
        help="Root directory containing one subdir per run (default: checkpoints/sweep).",
    )
    parser.add_argument(
        "--keys", nargs="*", default=None,
        help=(
            "Optional list of run keys to include "
            "(default: every immediate subdirectory of --sweep-root)."
        ),
    )
    parser.add_argument(
        "--rank-by", choices=("val_acc", "kendall"), default="val_acc",
        help="Metric used to pick the winner and sort the table.",
    )
    parser.add_argument(
        "--json", default=None,
        help="Optional path to write a machine-readable summary.",
    )
    args = parser.parse_args()

    sweep_root = Path(args.sweep_root)
    if not sweep_root.is_dir():
        raise SystemExit(f"Sweep root not found: {sweep_root}")

    if args.keys:
        keys = args.keys
    else:
        keys = sorted([p.name for p in sweep_root.iterdir() if p.is_dir()])

    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for key in keys:
        run_dir = sweep_root / key
        s = _summarize_run(key, run_dir)
        if s is None:
            skipped.append(key)
            continue
        rows.append(s)

    if not rows:
        raise SystemExit(
            f"No runs with a readable history.json under {sweep_root}.  "
            f"Skipped: {skipped}"
        )

    rank_key = {"val_acc": "val_acc", "kendall": "mean_kendall_tau"}[args.rank_by]

    print(f"\n── Sweep summary ({len(rows)} runs from {sweep_root}) ──\n")
    _print_table(rows, rank_by=rank_key)
    _recommend(rows, rank_by=rank_key)

    if skipped:
        print(f"Skipped (no history.json): {skipped}\n")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"Wrote machine-readable summary to {args.json}")


if __name__ == "__main__":
    main()
