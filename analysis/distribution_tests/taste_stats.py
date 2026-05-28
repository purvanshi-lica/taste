"""
Compute per-dimension statistics for TASTE CSVs in the (p=4, R=5)
fixed-panel framework.

A "sample" in TASTE is one prompt: 5 evaluators each rank the 4 models.
Each prompt yields:
  - 1 average pairwise Kendall tau value T
  - C(5,2) = 10 individual pairwise tau values
  - C(4,2) = 6 majority-vote probabilities  max(k/5, 1 - k/5)
  - 1 binary cycle indicator (majority-vote Condorcet cycle)

The 9 TASTE dimension CSVs each have 80 prompts -> 80 samples.

Output: one JSON + npz per dimension under ``stats/<slug>/``.
"""

import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from generate_reference_stats import _sample_stats  # noqa: E402

# Masked ranking CSVs are fetched into the repo data/ dir; taste_stats.py sits at
# analysis/distribution_tests/, two levels below the repo root. Override: TASTE_DATA_DIR.
DATA_DIR = Path(os.environ.get("TASTE_DATA_DIR", str(HERE.parents[1] / "data")))

MODELS = ["FLUX.2 [max]", "GPT Image 1.5", "Nano Banana 2", "Seedream 5.0 Lite"]

DIMENSIONS = [
    # slug, display, group, csv filename
    ("aesthetics_preference",    "UI+Ad Preference (holistic)", "Aesthetics",
     "aesthetics_%E2%80%94_eval_66%3A_ui%2Ba_preference_2026-04-24T0822.csv"),
    ("aesthetics_mood",          "Mood & Tone Match",           "Aesthetics",
     "aesthetics_%E2%80%94_eval_78%3A_mood_and_tone_match_2026-04-24T0822.csv"),
    ("aesthetics_visual_hier",   "Visual Hierarchy",            "Aesthetics",
     "aesthetics_%E2%80%94_eval_79%3A_visual_hierarchy_2026-04-24T0822.csv"),
    ("aesthetics_color_harmony", "Color Harmony",               "Aesthetics",
     "aesthetics_%E2%80%94_eval_80%3A_color_harmony_2026-04-24T0822.csv"),
    ("aesthetics_typography",    "Typography",                  "Aesthetics",
     "aesthetics_%E2%80%94_eval_81%3A_typography_2026-04-24T0823.csv"),
    ("descriptions_preference",  "Preference (holistic)",       "Descriptions",
     "descriptions_%E2%80%94_eval_72%3A_preference_2026-04-24T0823.csv"),
    ("descriptions_color_acc",   "Color Accuracy",              "Descriptions",
     "descriptions_%E2%80%94_eval_75%3A_color_accuracy_2026-04-24T0823.csv"),
    ("descriptions_spatial_acc", "Spatial Accuracy",            "Descriptions",
     "descriptions_%E2%80%94_eval_76%3A_spatial_accuracy_2026-04-24T0823.csv"),
    ("descriptions_typography",  "Typography",                  "Descriptions",
     "descriptions_%E2%80%94_eval_77%3A_typography_2026-04-24T0823.csv"),
]


def load_rank_tensor(csv_path):
    """Load a TASTE dimension CSV as a (n_prompts, R=5, p=4) rank tensor.

    Evaluator order is alphabetical; model order follows ``MODELS``.
    """
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=["model", "rank", "prompt_id", "evaluator"])
    # Validate per-(prompt, evaluator) groups have exactly ranks {1,2,3,4}
    groups = df.groupby(["prompt_id", "evaluator"])
    valid_mask = groups.apply(
        lambda x: sorted(x["rank"].tolist()) == [1, 2, 3, 4],
        include_groups=False,
    )
    valid_idx = valid_mask[valid_mask].index
    df = df.set_index(["prompt_id", "evaluator"]).loc[valid_idx].reset_index()

    prompts = sorted(df["prompt_id"].unique())
    evaluators = sorted(df["evaluator"].unique())
    tensor = np.empty((len(prompts), len(evaluators), len(MODELS)), dtype=np.int64)
    for pi, pid in enumerate(prompts):
        for ei, ev in enumerate(evaluators):
            sub = df[(df["prompt_id"] == pid) & (df["evaluator"] == ev)]
            rank_map = dict(zip(sub["model"], sub["rank"]))
            for mi, m in enumerate(MODELS):
                tensor[pi, ei, mi] = rank_map[m]
    return tensor, prompts, evaluators


def compute_stats(tensor, evaluators):
    """Run _sample_stats on every prompt.

    Returns dict with keys T, pair_tau, pmaj, cycle (numpy arrays) plus
    ``pair_tau_labels`` -- a list of (evaluator_i, evaluator_j) strings
    aligned with the columns of pair_tau.
    """
    n_prompts = tensor.shape[0]
    R = tensor.shape[1]
    Ts = np.empty(n_prompts, dtype=np.float64)
    taus = np.empty((n_prompts, 10), dtype=np.float64)
    pmajs = np.empty((n_prompts, 6), dtype=np.float64)
    cycs = np.empty(n_prompts, dtype=np.int64)
    for i in range(n_prompts):
        T, tau, pmaj, cyc = _sample_stats(tensor[i])
        Ts[i] = T
        taus[i] = tau
        pmajs[i] = pmaj
        cycs[i] = cyc
    # Column labels: pair indices from _sample_stats use np.triu_indices(R, k=1)
    # which is the same as itertools.combinations(range(R), 2) in lexicographic
    # order.
    pair_labels = [(evaluators[i], evaluators[j])
                   for i, j in combinations(range(R), 2)]
    return {"T": Ts, "pair_tau": taus, "pmaj": pmajs, "cycle": cycs,
            "pair_tau_labels": pair_labels}


def main():
    out_root = HERE / "stats"
    out_root.mkdir(exist_ok=True)

    summary_rows = []
    for slug, display, group, csv_name in DIMENSIONS:
        csv_path = DATA_DIR / csv_name
        tensor, prompts, evaluators = load_rank_tensor(str(csv_path))
        stats = compute_stats(tensor, evaluators)
        out_dir = out_root / slug
        out_dir.mkdir(exist_ok=True)
        # pair_tau_labels is a list of tuples; save each column as a
        # separate label array for compatibility with npz.
        save_kwargs = {k: v for k, v in stats.items()
                       if k != "pair_tau_labels"}
        save_kwargs["prompts"] = np.asarray(prompts)
        save_kwargs["evaluators"] = np.asarray(evaluators)
        save_kwargs["pair_labels_a"] = np.asarray(
            [a for a, _ in stats["pair_tau_labels"]])
        save_kwargs["pair_labels_b"] = np.asarray(
            [b for _, b in stats["pair_tau_labels"]])
        np.savez(out_dir / "stats.npz", **save_kwargs)
        summary = {
            "slug": slug,
            "display": display,
            "group": group,
            "csv": csv_name,
            "n_prompts": int(tensor.shape[0]),
            "T_mean": float(stats["T"].mean()),
            "T_median": float(np.median(stats["T"])),
            "T_sd": float(stats["T"].std()),
            "pair_tau_mean": float(stats["pair_tau"].mean()),
            "pmaj_mean": float(stats["pmaj"].mean()),
            "cycle_rate": float(stats["cycle"].mean()),
        }
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        summary_rows.append(summary)
        print(f"{slug:28s} n={tensor.shape[0]:3d} "
              f"T_med={summary['T_median']:+.3f} "
              f"pairtau_mean={summary['pair_tau_mean']:+.3f} "
              f"pmaj_mean={summary['pmaj_mean']:.3f} "
              f"cycle={summary['cycle_rate']:.3f}")

    with open(out_root / "all_summaries.json", "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"\nSaved per-dimension stats under {out_root}/")


if __name__ == "__main__":
    main()
