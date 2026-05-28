"""Aggregate VLM-judge results into per-(model, criterion) accuracy.

Reads `results/{model_name}/{criterion}.jsonl` and produces a JSON
summary plus a per-model markdown report with:
  - pair-prediction accuracy: VLM verdict matches human-majority
  - coverage: fraction of pairs the VLM gave a parseable A/B verdict
  - error rate: fraction of pairs with failed inference (image fetch
    or model error)

Per-criterion accuracy is reported, and a Spearman correlation between
the VLM-aggregated rank and human-consensus rank is computed for each
prompt by counting per-pair wins.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
REPORT_DIR = HERE / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _load(model_dir: Path):
    """Load all per-criterion result jsonls for one model. Returns
    nested dict {criterion: list[record]}."""
    out = {}
    for f in sorted(model_dir.glob("*.jsonl")):
        recs = []
        with open(f) as fh:
            for line in fh:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        out[f.stem] = recs
    return out


def per_criterion_metrics(records: list) -> dict:
    n_total = len(records)
    if n_total == 0:
        return {"n": 0}
    n_ok = sum(1 for r in records if r.get("ok"))
    n_parsed = sum(1 for r in records if r.get("verdict") in ("A", "B"))
    n_correct = sum(
        1 for r in records
        if r.get("verdict") in ("A", "B")
        and r.get("verdict") == r.get("human_majority")
    )
    # Pairs where human_majority == "tie" are excluded from accuracy
    n_eligible = sum(
        1 for r in records
        if r.get("verdict") in ("A", "B")
        and r.get("human_majority") in ("A", "B")
    )
    n_correct_eligible = sum(
        1 for r in records
        if r.get("verdict") in ("A", "B")
        and r.get("human_majority") in ("A", "B")
        and r.get("verdict") == r.get("human_majority")
    )

    # Per-prompt rank correlation
    by_prompt = {}
    for r in records:
        if r.get("verdict") not in ("A", "B"):
            continue
        pid = r["prompt_id"]
        a, b = r["image_a_model"], r["image_b_model"]
        # Maintain win counts per prompt
        d = by_prompt.setdefault(pid, {"vlm": {}, "human": {}})
        winner_vlm = a if r["verdict"] == "A" else b
        d["vlm"][winner_vlm] = d["vlm"].get(winner_vlm, 0) + 1
        if r.get("human_majority") in ("A", "B"):
            winner_h = a if r["human_majority"] == "A" else b
            d["human"][winner_h] = d["human"].get(winner_h, 0) + 1

    rhos = []
    for pid, d in by_prompt.items():
        if len(d["vlm"]) < 2 or len(d["human"]) < 2:
            continue
        models = sorted(set(d["vlm"]) | set(d["human"]))
        vlm_score = np.array([d["vlm"].get(m, 0) for m in models])
        h_score = np.array([d["human"].get(m, 0) for m in models])
        if vlm_score.std() == 0 or h_score.std() == 0:
            continue
        rho, _ = spearmanr(vlm_score, h_score)
        if not np.isnan(rho):
            rhos.append(float(rho))

    mean_latency = float(np.mean(
        [r.get("latency_s", 0) for r in records if r.get("ok")]
    )) if n_ok > 0 else 0.0

    return {
        "n": n_total,
        "n_ok": n_ok,
        "n_parsed_AB": n_parsed,
        "coverage": round(n_parsed / n_total, 4),
        "error_rate": round(1 - n_ok / n_total, 4),
        "accuracy_strict": (
            round(n_correct / n_total, 4) if n_total else None
        ),
        "accuracy_excluding_human_ties": (
            round(n_correct_eligible / n_eligible, 4) if n_eligible else None
        ),
        "n_eligible_pairs": n_eligible,
        "spearman_rho_per_prompt_mean": (
            round(float(np.mean(rhos)), 4) if rhos else None
        ),
        "spearman_rho_per_prompt_n": len(rhos),
        "mean_latency_s": round(mean_latency, 3),
    }


def report_one_model(model_dir: Path) -> dict:
    name = model_dir.name
    records_by_crit = _load(model_dir)
    out = {"model": name, "criteria": {}}
    for crit, recs in records_by_crit.items():
        out["criteria"][crit] = per_criterion_metrics(recs)
    # Aggregate over criteria
    accs = [c.get("accuracy_excluding_human_ties") for c in out["criteria"].values()
            if c.get("accuracy_excluding_human_ties") is not None]
    rhos = [c.get("spearman_rho_per_prompt_mean") for c in out["criteria"].values()
            if c.get("spearman_rho_per_prompt_mean") is not None]
    out["overall"] = {
        "macro_accuracy": round(float(np.mean(accs)), 4) if accs else None,
        "macro_spearman": round(float(np.mean(rhos)), 4) if rhos else None,
    }
    return out


def render_markdown(reports):
    lines = [
        "# VLM-as-judge results on TASTE",
        "",
        "Per-(model, criterion) accuracy of VLM verdicts against the "
        "5-rater human-majority winner on each pair.  Pairs whose "
        "human label is a tie are excluded from the accuracy denominator.",
        "",
        "## Overall (macro-averaged across 9 criteria)",
        "",
        "| Model | macro acc | macro Spearman | mean latency (s) |",
        "|---|---|---|---|",
    ]
    for r in sorted(reports, key=lambda r: -(r["overall"]["macro_accuracy"] or 0)):
        lat_vals = [c.get("mean_latency_s", 0)
                    for c in r["criteria"].values()]
        mean_lat = np.mean(lat_vals) if lat_vals else 0.0
        lines.append(
            f"| {r['model']} | "
            f"{r['overall']['macro_accuracy']} | "
            f"{r['overall']['macro_spearman']} | "
            f"{mean_lat:.2f} |"
        )
    lines.append("")
    lines.append("## Per-criterion accuracy")
    lines.append("")
    crit_set = sorted({c for r in reports for c in r["criteria"].keys()})
    header = "| Model | " + " | ".join(crit_set) + " |"
    lines.append(header)
    lines.append("|---" * (len(crit_set) + 1) + "|")
    for r in sorted(reports, key=lambda r: -(r["overall"]["macro_accuracy"] or 0)):
        cells = [r["model"]]
        for c in crit_set:
            v = r["criteria"].get(c, {}).get("accuracy_excluding_human_ties")
            cells.append("--" if v is None else f"{v:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    if not RESULTS.exists():
        print("No results directory yet. Run run_vlm_judge.py first.")
        return
    reports = []
    for model_dir in sorted(RESULTS.iterdir()):
        if not model_dir.is_dir():
            continue
        rep = report_one_model(model_dir)
        reports.append(rep)
    summary_path = REPORT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(reports, f, indent=2)
    md_path = REPORT_DIR / "summary.md"
    with open(md_path, "w") as f:
        f.write(render_markdown(reports))
    print(f"wrote {summary_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
