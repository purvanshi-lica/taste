"""
For each TASTE dimension: produce the 4-panel main figure, the 2x5
per-evaluator-pair grid, JSON of test statistics, and a per-dimension
disagreement-examples markdown fragment.

Usage:
    python run_dimension.py [--slug <slug>] [--top-n 5]

Default: process all 9 dimensions defined in taste_stats.DIMENSIONS.

Outputs under  figures/<slug>/:
    - main.png                4-panel histogram figure
    - pair_reviewers.png      2x5 per-pair tau histograms
    - tests.json              chi2/KS/binomial stats per (statistic, comparator)
    - disagreement.md         top-N most-disagreed-on prompts with images
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from taste_stats import DIMENSIONS, load_rank_tensor, compute_stats, \
    DATA_DIR, MODELS  # noqa: E402
from distribution_plots import (  # noqa: E402
    plot_main_figure, plot_pair_reviewer_grid, run_tests, cycle_null_rate,
)

MODEL_SHORT = {"FLUX.2 [max]": "FLUX.2", "GPT Image 1.5": "GPT-Img",
               "Nano Banana 2": "NanoBan", "Seedream 5.0 Lite": "Seedream"}

REF_NAMES = ["Sushi", "MovieLens", "HPSv2-test"]
REF_FILENAMES = {"Sushi": "sushi", "MovieLens": "movielens",
                 "HPSv2-test": "hpsv2"}


def load_refs():
    refs_dir = HERE / "refs"
    out = {}
    for name in REF_NAMES:
        slug = REF_FILENAMES[name]
        out[name] = {
            "T":        np.load(refs_dir / f"{slug}_T.npy"),
            "pair_tau": np.load(refs_dir / f"{slug}_pairtau.npy"),
            "pmaj":     np.load(refs_dir / f"{slug}_pmaj.npy"),
            "cycle":    np.load(refs_dir / f"{slug}_cycle.npy"),
        }
    return out


def _resize_cloudinary(url, width=280):
    """Insert Cloudinary width+format transform into image URLs."""
    marker = "/image/upload/"
    i = url.find(marker)
    if i == -1:
        return url
    head = url[: i + len(marker)]
    tail = url[i + len(marker):]
    if tail.split("/")[0].startswith(("w_", "c_", "f_", "q_")):
        return url
    return f"{head}w_{width},c_fit,q_auto,f_auto/{tail}"


# Section headers we recognize when splitting the structured prompt column.
_PROMPT_SECTIONS = ["User Intent:", "Aesthetics:", "Description:"]


def _split_prompt(text):
    """Split the raw `prompt` column into a dict {section -> body}.

    The TASTE CSVs use a fixed format with section headers like
    ``User Intent:`` and ``Aesthetics:`` (or ``Description:`` for the
    descriptions cohort).  Returns an OrderedDict-like list of (header,
    body) tuples in the original order.  Anything before the first
    recognized header is keyed as "Other".
    """
    text = str(text)
    # Find each header occurrence
    cuts = []
    for h in _PROMPT_SECTIONS:
        idx = 0
        while True:
            j = text.find(h, idx)
            if j == -1:
                break
            cuts.append((j, h))
            idx = j + len(h)
    cuts.sort()
    if not cuts:
        return [("Prompt", text.strip())]
    out = []
    if cuts[0][0] > 0:
        head = text[: cuts[0][0]].strip()
        if head:
            out.append(("Other", head))
    for k, (start, header) in enumerate(cuts):
        body_start = start + len(header)
        body_end = cuts[k + 1][0] if k + 1 < len(cuts) else len(text)
        body = text[body_start:body_end].strip()
        out.append((header.rstrip(":"), body))
    return out


def _evaluator_alignment(prompt_idx, stats, evaluators):
    """For a given prompt and each evaluator, mean tau against the other 4.

    A higher value means this evaluator is more aligned with the
    consensus on this specific prompt.  Range [-1, +1].
    """
    pair_labels = stats["pair_tau_labels"]
    taus_for_prompt = stats["pair_tau"][prompt_idx]            # (10,)
    by_eval = {e: [] for e in evaluators}
    for col, (e1, e2) in enumerate(pair_labels):
        by_eval[e1].append(taus_for_prompt[col])
        by_eval[e2].append(taus_for_prompt[col])
    return {e: float(np.mean(vs)) for e, vs in by_eval.items()}


def _hallucination_counts(pid):
    """For holistic dims (R1: 608-687, R2: 851-930), look up the per-image
    hallucination-flag counts.  Returns dict {model_name: (n_minor_or_major,
    n_major)} or empty dict if no hallucination data covers this pid.
    """
    if 608 <= pid <= 687:
        path = DATA_DIR / "hallucination_flags_2026-04-24T0822.csv"
    elif 851 <= pid <= 930:
        path = DATA_DIR / "v2_hallucination_flags_2026-04-24T0823.csv"
    else:
        return {}
    if not path.exists():
        return {}
    try:
        h = pd.read_csv(path)
    except Exception:
        return {}
    sub = h[h["prompt_id"] == pid]
    if sub.empty:
        return {}
    out = {}
    for model in MODELS:
        ms = sub[sub["model"] == model]
        if ms.empty:
            continue
        n_any  = int((ms["hallucination_value"] >= 1).sum())
        n_maj  = int((ms["hallucination_value"] == 2).sum())
        n_tot  = int(len(ms))
        out[model] = {"any": n_any, "major": n_maj, "total": n_tot}
    return out


def write_disagreement_md(slug, display, csv_path, tensor, prompts, evaluators,
                           stats, top_n, out_path):
    """Render top-N most-disagreed-on prompts ranked by ascending per-prompt T.

    Includes:
      - Full structured prompt split into User Intent / Aesthetics-or-Description
      - 4 model output thumbnails (Cloudinary 280px, link to full-res)
      - Per-evaluator rank matrix
      - Per-evaluator "consensus alignment" = mean tau against other 4
        evaluators on this specific prompt (proxy for confidence in their
        own ranking matching the group)
      - For holistic dims (608-687, 851-930): per-image hallucination flag
        count (X/5 evaluators flagged any-hallucination, Y/5 Major-only)
    """
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=["model", "rank", "prompt_id", "evaluator"])

    order = np.argsort(stats["T"])
    lines = [f"# Disagreement examples — {display}", "",
             f"Top {top_n} prompts with the lowest per-prompt $T$ "
             f"(= mean pairwise Kendall $\\tau$ across the 10 evaluator pairs). "
             f"Lower $T$ = reviewers disagree more.", "",
             "Each row shows: full structured prompt (User Intent / "
             "Aesthetics-or-Description), the four model outputs (280px "
             "Cloudinary thumbs, click for full-res), the per-evaluator rank "
             "matrix, and a **consensus alignment** score per evaluator "
             "(mean $\\tau$ against the other 4 on this prompt; high = aligned "
             "with group, low or negative = outlier on this prompt). For the "
             "two holistic-preference dimensions where hallucination data "
             "exists (prompt IDs 608-687 and 851-930), per-image "
             "hallucination flag counts are also shown.", ""]

    for rank_idx, prompt_idx in enumerate(order[:top_n], 1):
        pid = int(prompts[prompt_idx])
        T_val = float(stats["T"][prompt_idx])
        cyc = bool(stats["cycle"][prompt_idx])
        pmaj_mean = float(stats["pmaj"][prompt_idx].mean())

        grp = df[df["prompt_id"] == pid]
        prompt_text = grp["prompt"].iloc[0]
        sections = _split_prompt(prompt_text)

        first_ev = sorted(grp["evaluator"].unique())[0]
        urls = {}
        for m in MODELS:
            sub = grp[(grp["evaluator"] == first_ev) & (grp["model"] == m)]
            urls[m] = sub["model_output_image_url"].iloc[0] if len(sub) else ""

        alignment = _evaluator_alignment(prompt_idx, stats, evaluators)
        halluc = _hallucination_counts(pid)

        cyc_flag = " **(Condorcet cycle)**" if cyc else ""
        lines.append(f"## #{rank_idx}  prompt {pid}   "
                     f"$T={T_val:+.3f}$,  mean $p_{{\\max}}={pmaj_mean:.3f}${cyc_flag}")
        lines.append("")
        for header, body in sections:
            lines.append(f"**{header}.**  {body}")
            lines.append("")

        # Image grid with hallucination flag annotation when available
        lines.append("<table><tr>")
        for m in MODELS:
            short = MODEL_SHORT[m]
            url = urls[m]
            thumb = _resize_cloudinary(url)
            halluc_note = ""
            if m in halluc:
                hh = halluc[m]
                halluc_note = (f"<br><sub>halluc: {hh['any']}/{hh['total']} any · "
                               f"{hh['major']}/{hh['total']} major</sub>")
            if url:
                lines.append(
                    f'<td align="center">'
                    f'<sub><b>{short}</b></sub><br>'
                    f'<a href="{url}"><img src="{thumb}" width="280" alt="{short}"></a>'
                    f'{halluc_note}</td>'
                )
            else:
                lines.append(f'<td align="center"><sub>{short}</sub><br><em>(no image)</em></td>')
        lines.append("</tr></table>")
        lines.append("")

        # Rank + alignment matrix
        lines.append("**Per-evaluator ranks (1 = best, 4 = worst) and consensus "
                     "alignment $\\bar\\tau_{\\text{others}}$ on this prompt:**")
        lines.append("")
        header = ("| Evaluator | "
                  + " | ".join(MODEL_SHORT[m] for m in MODELS)
                  + " | $\\bar\\tau_{\\text{others}}$ |")
        sep = "|---" * (len(MODELS) + 2) + "|"
        lines.append(header)
        lines.append(sep)
        # Sort evaluators by alignment ascending so outliers appear at top
        sorted_evs = sorted(evaluators, key=lambda e: alignment[e])
        for ev in sorted_evs:
            ei = list(evaluators).index(ev)
            ranks_row = tensor[prompt_idx, ei].tolist()
            align = alignment[ev]
            lines.append(f"| {ev} | "
                         + " | ".join(str(r) for r in ranks_row)
                         + f" | **{align:+.3f}** |")
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def process_dimension(slug, display, group, csv_name, refs_dict,
                      cyc_null_rt, top_n, out_root):
    csv_path = DATA_DIR / csv_name
    tensor, prompts, evaluators = load_rank_tensor(str(csv_path))
    stats = compute_stats(tensor, evaluators)

    out_dir = out_root / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    tests = run_tests(stats, refs_dict, cyc_null_rt=cyc_null_rt)

    # Strip large label list before JSON dump
    pair_labels = stats.pop("pair_tau_labels")
    tests_payload = dict(tests)                                 # shallow copy
    tests_payload["_pair_labels"] = [list(p) for p in pair_labels]
    with open(out_dir / "tests.json", "w") as f:
        json.dump(tests_payload, f, indent=2, default=float)
    stats["pair_tau_labels"] = pair_labels                      # restore for plotting

    plot_main_figure(
        stats, refs_dict, tests,
        out_path=str(out_dir / "main.png"),
        title=f"{display}  ({group})",
    )
    plot_pair_reviewer_grid(
        stats,
        out_path=str(out_dir / "pair_reviewers.png"),
        title=f"{display}  ({group}) — pairwise $\\tau$ by evaluator pair "
              f"(R={len(evaluators)}, 10 pairs, 80 prompts each)",
    )
    write_disagreement_md(
        slug, display, str(csv_path), tensor, prompts, evaluators,
        stats, top_n, str(out_dir / "disagreement.md"),
    )

    # Quick console summary
    print(f"  {slug:28s}  T median={np.median(stats['T']):+.3f}  "
          f"GOF vs null p={tests['T']['null']['p']:.2e}  "
          f"cycle={stats['cycle'].mean():.3f}  "
          f"wrote: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", default=None,
                        help="Process a single dimension slug (default: all).")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Number of disagreement exemplars per dimension.")
    parser.add_argument("--out-dir", default=str(HERE / "figures"),
                        help="Root directory for per-dimension outputs.")
    args = parser.parse_args()

    refs_dict = load_refs()
    print("Computing cycle null rate (MC)...")
    cyc_null_rt = cycle_null_rate()
    print(f"  null rate = {cyc_null_rt:.4f}")

    dims = [d for d in DIMENSIONS if args.slug is None or d[0] == args.slug]
    if not dims:
        raise SystemExit(f"No dimension matches --slug={args.slug}")
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for slug, display, group, csv_name in dims:
        process_dimension(slug, display, group, csv_name, refs_dict,
                          cyc_null_rt, args.top_n, out_root)

    print(f"\nAll outputs under {out_root}/")


if __name__ == "__main__":
    main()
