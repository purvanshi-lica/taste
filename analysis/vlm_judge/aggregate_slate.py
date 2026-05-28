"""Paper-grade aggregation across the full TASTE VLM-judge slate.

Produces `reports/slate_summary.{json,md}`.  Different from
`analyze_results.py` in three ways:

  1. **Only canonical full-eval directories are included.**  Smoke
     and methodology-probe runs (single-image, fewshot variants,
     paraphrase scaling probes, etc.) live in `results/` but are
     filtered out here so the published artifact matches the paper.

  2. **VLM rows pair `{name}-FULL` and `{name}-FULL-flip` and
     compute MT-Bench S1** (Zheng et al. 2023) plus position-bias
     rate and discrim-only accuracy.

  3. **Scorer rows** (deterministic models) are processed without
     flip pairing.

The slate is hard-coded in `SLATE` below so adding or removing
models is a one-line edit.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
REPORT_DIR = HERE / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# (display_name, kind, primary_dir, flip_dir-or-None)
SLATE = [
    ("Qwen3-VL-4B-Instruct",  "vlm",
     "Qwen3-VL-4B-Instruct-FULL",  "Qwen3-VL-4B-Instruct-FULL-flip"),
    ("Qwen3-VL-8B-Instruct",  "vlm",
     "Qwen3-VL-8B-Instruct-FULL",  "Qwen3-VL-8B-Instruct-FULL-flip"),
    ("Qwen3-VL-32B-Instruct", "vlm",
     "Qwen3-VL-32B-Instruct-FULL", "Qwen3-VL-32B-Instruct-FULL-flip"),
    ("Gemma-3-27B-it",        "vlm",
     "Gemma-3-27B-it-FULL",        "Gemma-3-27B-it-FULL-flip"),
    ("Kimi-VL-A3B-Instruct",  "vlm",
     "Kimi-VL-A3B-Instruct-FULL", "Kimi-VL-A3B-Instruct-FULL-flip"),
    ("InternVL3_5-14B",  "vlm",
     "InternVL3_5-14B-FULL", "InternVL3_5-14B-FULL-flip"),
    ("HPSv2.1",            "scorer", "HPSv2.1",            None),
    ("PickScore-v1",       "scorer", "PickScore-v1",       None),
    ("LAION-Aesthetic-V2", "scorer", "LAION-Aesthetic-V2", None),
]

CRITERIA = [
    "aesthetics_color_harmony", "aesthetics_mood",
    "aesthetics_preference", "aesthetics_typography",
    "aesthetics_visual_hier",
    "descriptions_color_acc", "descriptions_preference",
    "descriptions_spatial_acc", "descriptions_typography",
]


def _load(model_dir: Path):
    out = {}
    for f in sorted(model_dir.glob("*.jsonl")):
        with open(f) as fh:
            recs = [json.loads(line) for line in fh if line.strip()]
        out[f.stem] = recs
    return out


def _canon_key(r):
    if r.get("flipped"):
        return (r["prompt_id"], r["image_b_model"], r["image_a_model"])
    return (r["prompt_id"], r["image_a_model"], r["image_b_model"])


def _canon_pred(r):
    if r.get("flipped"):
        return {"A": "B", "B": "A"}.get(r["verdict"], r["verdict"])
    return r["verdict"]


def eval_vlm(nf_recs, fl_recs):
    """Returns dict with S1, discrim_only, pos_bias, parse_cov,
    per_paraphrase_std for one VLM/criterion."""
    n_total = len(nf_recs)
    if n_total == 0:
        return {}
    nf_by = {_canon_key(r): r for r in nf_recs}
    fl_by = {_canon_key(r): r for r in fl_recs}
    keys = sorted(set(nf_by) & set(fl_by))

    n_consistent = n_tie = n_correct = eligible = 0
    n_parsed_nf = 0
    for k in keys:
        ch = nf_by[k]["human_majority"]
        p_nf = _canon_pred(nf_by[k])
        if p_nf in ("A", "B"):
            n_parsed_nf += 1
        if ch not in ("A", "B"):
            continue
        eligible += 1
        p_fl = _canon_pred(fl_by[k])
        if p_nf == p_fl and p_nf in ("A", "B"):
            n_consistent += 1
            if p_nf == ch:
                n_correct += 1
        else:
            n_tie += 1

    if eligible == 0:
        return {}
    S1 = (n_correct + 0.5 * n_tie) / eligible
    discrim = n_correct / max(n_consistent, 1)
    pos_bias = n_tie / eligible

    # Per-paraphrase accuracy (no-flip only)
    by_para = defaultdict(list)
    for r in nf_recs:
        if r["verdict"] in ("A", "B") and r["human_majority"] in ("A", "B"):
            by_para[r.get("paraphrase_idx", 0)].append(
                int(r["verdict"] == r["human_majority"])
            )
    para_accs = [np.mean(v) for v in by_para.values() if len(v) >= 5]
    para_std = float(np.std(para_accs)) if len(para_accs) > 1 else None

    return {
        "n_eligible": eligible,
        "S1": round(S1, 4),
        "discrim_only": round(discrim, 4),
        "pos_bias": round(pos_bias, 4),
        "parse_cov": round(n_parsed_nf / len(keys), 4) if keys else None,
        "para_std": round(para_std, 4) if para_std is not None else None,
        "n_pairs": len(keys),
    }


def eval_scorer(recs):
    elig = [r for r in recs
            if r["verdict"] in ("A", "B")
            and r["human_majority"] in ("A", "B")]
    if not elig:
        return {}
    return {
        "n_eligible": len(elig),
        "accuracy": round(
            sum(1 for r in elig if r["verdict"] == r["human_majority"])
            / len(elig), 4
        ),
        "n_pairs": len(recs),
    }


def aggregate():
    rows = []
    for display, kind, primary, flip in SLATE:
        primary_path = RESULTS / primary
        if not primary_path.exists():
            print(f"  [skip] missing dir {primary}")
            continue
        recs_by_crit = _load(primary_path)
        if kind == "vlm":
            flip_path = RESULTS / flip
            if not flip_path.exists():
                print(f"  [skip] missing flip dir {flip}")
                continue
            flip_recs_by_crit = _load(flip_path)
            per_crit = {}
            for c in CRITERIA:
                if c not in recs_by_crit or c not in flip_recs_by_crit:
                    continue
                per_crit[c] = eval_vlm(recs_by_crit[c], flip_recs_by_crit[c])
            # Macro (mean across criteria)
            S1s = [v["S1"] for v in per_crit.values()]
            disc = [v["discrim_only"] for v in per_crit.values()]
            pbs = [v["pos_bias"] for v in per_crit.values()]
            paras = [v["para_std"] for v in per_crit.values() if v["para_std"] is not None]
            n_total = sum(v["n_pairs"] for v in per_crit.values())
            rows.append({
                "model": display, "kind": "vlm",
                "n_pairs_total": n_total,
                "macro_S1": round(float(np.mean(S1s)), 4),
                "macro_discrim_only": round(float(np.mean(disc)), 4),
                "macro_pos_bias": round(float(np.mean(pbs)), 4),
                "macro_para_std": round(float(np.mean(paras)), 4) if paras else None,
                "per_criterion": per_crit,
            })
        else:  # scorer
            per_crit = {}
            for c in CRITERIA:
                if c not in recs_by_crit:
                    continue
                per_crit[c] = eval_scorer(recs_by_crit[c])
            accs = [v["accuracy"] for v in per_crit.values() if "accuracy" in v]
            n_total = sum(v["n_pairs"] for v in per_crit.values() if "n_pairs" in v)
            rows.append({
                "model": display, "kind": "scorer",
                "n_pairs_total": n_total,
                "macro_accuracy": round(float(np.mean(accs)), 4) if accs else None,
                "per_criterion": per_crit,
            })
    return rows


def render_md(rows):
    lines = []
    lines.append("# TASTE VLM-judge slate — paper-grade summary")
    lines.append("")
    lines.append("Generated by `aggregate_slate.py`.  Includes only canonical")
    lines.append("full-eval directories (`*-FULL` for VLMs paired with `*-FULL-flip`;")
    lines.append("single dir for deterministic scorers).  Methodology and smoke runs")
    lines.append("(single-image diagnostic, prompt-engineering variants, fewshot probes)")
    lines.append("are deliberately excluded.")
    lines.append("")
    lines.append("## Headline macro table")
    lines.append("")
    lines.append("| Model | Class | n (pairs) | Macro S1 | Pos-bias | Discrim-only | Paraphrase σ |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        if r["kind"] == "vlm":
            lines.append(
                f"| {r['model']} | VLM-as-judge | {r['n_pairs_total']} "
                f"| {r['macro_S1']:.3f} "
                f"| {r['macro_pos_bias']:.3f} "
                f"| {r['macro_discrim_only']:.3f} "
                f"| {r['macro_para_std']:.3f} |"
            )
        else:
            lines.append(
                f"| {r['model']} | scorer (det.) | {r['n_pairs_total']} "
                f"| {r['macro_accuracy']:.3f} | — | — | — |"
            )
    lines.append("")
    lines.append("Macro is mean across the 9 criteria, equal-weighted.  ")
    lines.append("**MT-Bench S1** (Zheng et al. 2023): order-inconsistent pairs treated as ")
    lines.append("ties at 0.5.  **Pos-bias**: fraction of pairs where the model's verdict ")
    lines.append("agrees with itself post-flip (i.e., the order-determined fraction).  ")
    lines.append("**Discrim-only**: agreement-with-designers on the consistent fraction.  ")
    lines.append("**Paraphrase σ**: standard deviation of per-criterion accuracy across the ")
    lines.append("8 question paraphrases.")
    lines.append("")

    # Per-criterion S1 table
    lines.append("## Per-criterion MT-Bench S1 accuracy")
    lines.append("")
    vlm_rows = [r for r in rows if r["kind"] == "vlm"]
    scorer_rows = [r for r in rows if r["kind"] == "scorer"]
    headers = ["Criterion"] + [r["model"] for r in vlm_rows + scorer_rows]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|---" * len(headers) + "|")
    for c in CRITERIA:
        cells = [c]
        for r in vlm_rows:
            v = r["per_criterion"].get(c, {})
            cells.append(f"{v['S1']:.3f}" if "S1" in v else "—")
        for r in scorer_rows:
            v = r["per_criterion"].get(c, {})
            cells.append(f"{v['accuracy']:.3f}" if "accuracy" in v else "—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Per-criterion position-bias for VLMs only
    lines.append("## Per-criterion position-bias rate (VLM only)")
    lines.append("")
    headers = ["Criterion"] + [r["model"] for r in vlm_rows]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|---" * len(headers) + "|")
    for c in CRITERIA:
        cells = [c]
        for r in vlm_rows:
            v = r["per_criterion"].get(c, {})
            cells.append(f"{v['pos_bias']:.3f}" if "pos_bias" in v else "—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main():
    rows = aggregate()
    json_path = REPORT_DIR / "slate_summary.json"
    md_path = REPORT_DIR / "slate_summary.md"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    with open(md_path, "w") as f:
        f.write(render_md(rows))
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
