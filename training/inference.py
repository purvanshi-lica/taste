"""
Inference + evaluation for multi-dimension TASTE.

Loads a trained checkpoint (per-dimension MLP heads + optional
hallucination head + optional LoRA adapter) or a raw VLM baseline
(no heads → cosine-similarity score), and exposes:

* a programmatic ``HPSContraScorer`` class with ``score``, ``compare``,
  ``rank_images`` and ``hallucination`` methods.
* CLI subcommands ``score``, ``compare``, ``batch``, ``halluc``, ``eval``.

Per-dimension scoring requires a ``--dimension`` argument; the available
dimension slugs are listed in ``meta.json`` of the checkpoint.  Common
dimension slugs (after the data refactor):
``preference``, ``typography``, ``color_harmony``, ``mood_and_color_tone``,
``visual_hierarchy``, ``color_accuracy``, ``spatial_accuracy``.

Usage:
    python inference.py score --checkpoint ckpt/best \
        --dimension preference --prompt "..." --image photo.jpg

    python inference.py compare --checkpoint ckpt/best \
        --dimension typography \
        --prompt "..." --image-a a.jpg --image-b b.jpg

    python inference.py halluc --checkpoint ckpt/best \
        --prompt "..." --image photo.jpg

    python inference.py eval --checkpoint ckpt/best \
        --val-csv data/battles_val.csv \
        --halluc-val-csv data/halluc_val.csv \
        --image-dir data/images
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from embedders import VLEmbedder


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------
class HPSContraScorer:
    """Load a trained multi-head checkpoint or a raw VLM baseline and score pairs.

    In trained mode the scorer dispatches to a per-dimension MLP head; in
    baseline mode (``baseline_model`` constructor arg) it falls back to
    cosine similarity between text and image embeddings, which is what the
    pre-trained Qwen3-VL/CLIP/SigLIP encoders give you out of the box.
    """

    def __init__(
        self,
        checkpoint_dir: str | None = None,
        device: str | None = None,
        baseline_model: str | None = None,
    ):
        if (checkpoint_dir is None) == (baseline_model is None):
            raise ValueError("Provide exactly one of checkpoint_dir or baseline_model")

        _device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.device = _device
        self.scorer = None  # MultiHeadScorer when trained; None for baseline
        self.halluc_head = None  # HallucinationHead if checkpoint has one
        self.dimensions: list[str] = []
        self.meta: dict = {}

        if baseline_model:
            self.embedder = VLEmbedder.from_pretrained(baseline_model, device=_device)
            self.embedder.model.eval()
            self.mode = "baseline"
            return

        ckpt = Path(checkpoint_dir)
        with open(ckpt / "meta.json") as f:
            self.meta = json.load(f)

        self.embedder = VLEmbedder.from_pretrained(
            self.meta["model_name"], device=_device
        )

        if self.meta.get("enable_lora") and (ckpt / "lora_adapter").exists():
            from peft import PeftModel
            print(f"Loading LoRA adapter from {ckpt / 'lora_adapter'} ...")
            self.embedder.model = PeftModel.from_pretrained(
                self.embedder.model, str(ckpt / "lora_adapter")
            )
        self.embedder.model.eval()

        heads_path = ckpt / "heads.pt"
        if heads_path.exists():
            from heads import MultiHeadScorer

            payload = torch.load(heads_path, map_location=_device, weights_only=True)
            self.scorer = MultiHeadScorer.from_checkpoint(payload).to(_device).eval()
            self.dimensions = list(self.scorer.dimensions)
            self.mode = "multi_head"
            print(f"Loaded {len(self.dimensions)} scoring heads: {self.dimensions}")
        else:
            self.mode = "lora_cosine"  # legacy: LoRA-only checkpoint, no heads

        halluc_path = ckpt / "halluc_head.pt"
        if halluc_path.exists():
            from heads import HallucinationHead

            payload = torch.load(halluc_path, map_location=_device, weights_only=True)
            self.halluc_head = (
                HallucinationHead.from_checkpoint(payload).to(_device).eval()
            )
            print("Loaded hallucination head")

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_text(self, prompt: str) -> torch.Tensor:
        return self.embedder.encode_text([prompt])

    @torch.no_grad()
    def encode_image(self, image_path: str) -> torch.Tensor:
        return self.embedder.encode_image([image_path])

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def _resolve_dimension(self, dimension: str | None) -> str | None:
        if self.mode != "multi_head":
            return None
        if dimension is None:
            if len(self.dimensions) == 1:
                return self.dimensions[0]
            raise ValueError(
                f"Checkpoint has {len(self.dimensions)} dimensions; "
                f"specify one of: {self.dimensions}"
            )
        if dimension not in self.dimensions:
            raise ValueError(
                f"Unknown dimension '{dimension}'. "
                f"Available: {self.dimensions}"
            )
        return dimension

    def _score_pair(
        self,
        text_emb: torch.Tensor,
        image_emb: torch.Tensor,
        dimension: str | None,
    ) -> torch.Tensor:
        if self.mode == "multi_head":
            assert dimension is not None
            return self.scorer.score(
                text_emb.to(torch.float32),
                image_emb.to(torch.float32),
                dimension,
            )
        return F.cosine_similarity(text_emb, image_emb, dim=-1)

    @torch.no_grad()
    def score(self, prompt: str, image_path: str, dimension: str | None = None) -> float:
        dim = self._resolve_dimension(dimension)
        text_emb = self.encode_text(prompt)
        img_emb = self.encode_image(image_path)
        return self._score_pair(text_emb, img_emb, dim).item()

    @torch.no_grad()
    def compare(
        self,
        prompt: str,
        image_a: str,
        image_b: str,
        dimension: str | None = None,
    ) -> dict:
        dim = self._resolve_dimension(dimension)
        text_emb = self.encode_text(prompt)
        img_a_emb = self.encode_image(image_a)
        img_b_emb = self.encode_image(image_b)
        score_a = self._score_pair(text_emb, img_a_emb, dim).item()
        score_b = self._score_pair(text_emb, img_b_emb, dim).item()
        return {
            "score_a": score_a,
            "score_b": score_b,
            "predicted_winner": "A" if score_a > score_b else "B",
            "margin": abs(score_a - score_b),
            "dimension": dim,
        }

    @torch.no_grad()
    def hallucination(self, prompt: str, image_path: str) -> dict:
        """Predict the probability that ``image_path`` hallucinates content
        not described by ``prompt``.

        Returns ``{"prob": float, "predicted_label": int, "logit": float}``.
        Raises ``RuntimeError`` if the checkpoint has no hallucination head.
        """
        if self.halluc_head is None:
            raise RuntimeError(
                "This checkpoint has no hallucination head; train with the "
                "halluc CSVs to enable it."
            )
        text_emb = self.encode_text(prompt).to(torch.float32)
        img_emb = self.encode_image(image_path).to(torch.float32)
        logit = self.halluc_head(text_emb, img_emb)
        prob = torch.sigmoid(logit).item()
        return {
            "logit": float(logit.item()),
            "prob": float(prob),
            "predicted_label": int(prob >= 0.5),
        }

    @torch.no_grad()
    def rank_images(
        self,
        prompt: str,
        image_paths: list[str],
        dimension: str | None = None,
    ) -> list[dict]:
        dim = self._resolve_dimension(dimension)
        text_emb = self.encode_text(prompt)
        results = []
        for path in image_paths:
            img_emb = self.encode_image(path)
            s = self._score_pair(text_emb, img_emb, dim).item()
            results.append({"image": path, "score": s})
        results.sort(key=lambda x: x["score"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1
        return results


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------
def _kendall_tau(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    concordant = discordant = 0
    ties_x = ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 and dy == 0:
                ties_x += 1
                ties_y += 1
            elif dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif (dx > 0 and dy > 0) or (dx < 0 and dy < 0):
                concordant += 1
            else:
                discordant += 1
    npairs = n * (n - 1) / 2
    denom_x = npairs - ties_x
    denom_y = npairs - ties_y
    if denom_x == 0 or denom_y == 0:
        return 0.0
    return (concordant - discordant) / (denom_x * denom_y) ** 0.5


def compute_eval_metrics(results: list[dict]) -> dict:
    """Compute pairwise accuracy + per-bucket accuracy + Kendall tau.

    ``results`` may span multiple dimensions; tau is computed per
    ``(dimension, prompt_id)`` group so different dimensions don't pollute
    each other's per-prompt rankings (they currently use disjoint prompt
    sets but this is the safer key regardless).
    """
    n = len(results)
    if n == 0:
        return {
            "accuracy": 0.0,
            "n": 0,
            "accuracy_by_agreement": {
                "unanimous": (0.0, 0), "majority": (0.0, 0), "split": (0.0, 0)
            },
            "mean_kendall_tau": 0.0,
            "n_prompts": 0,
        }
    n_correct = sum(1 for r in results if r["correct"])
    accuracy = n_correct / n

    buckets: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        bucket = r.get("agreement_bucket")
        if not bucket:
            agr = float(r.get("agreement", 1.0))
            # 5 evaluators ⇒ agreement ∈ {0.6, 0.8, 1.0}; thresholds chosen
            # so that 5-0 → unanimous, 4-1 → majority, 3-2 → split.
            bucket = (
                "unanimous" if agr >= 0.95
                else ("majority" if agr >= 0.7 else "split")
            )
        buckets[bucket].append(bool(r["correct"]))

    accuracy_by_agreement: dict[str, tuple[float, int]] = {}
    for label in ("unanimous", "majority", "split"):
        vals = buckets.get(label, [])
        accuracy_by_agreement[label] = (
            (sum(vals) / len(vals), len(vals)) if vals else (0.0, 0)
        )

    prompts: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in results:
        prompts[(r.get("dimension", ""), r.get("prompt_id", ""))].append(r)

    taus = []
    for _, battles in prompts.items():
        images: dict[str, dict] = {}
        for b in battles:
            for side in ("a", "b"):
                img = b[f"image_{side}"]
                if img not in images:
                    images[img] = {"model_scores": [], "human_ranks": []}
                images[img]["model_scores"].append(float(b[f"score_{side}"]))
                hr = b.get(f"mean_rank_{side}", b.get(f"rank_{side}"))
                if hr is None:
                    continue
                images[img]["human_ranks"].append(float(hr))

        if len(images) < 2:
            continue

        img_list = list(images.keys())
        model_avg = [
            sum(images[i]["model_scores"]) / len(images[i]["model_scores"])
            for i in img_list
        ]
        human_avg = [
            (sum(images[i]["human_ranks"]) / len(images[i]["human_ranks"]))
            if images[i]["human_ranks"] else 0.0
            for i in img_list
        ]
        # negate model_avg because lower human rank = better
        taus.append(_kendall_tau([-s for s in model_avg], human_avg))

    mean_tau = sum(taus) / len(taus) if taus else 0.0

    return {
        "accuracy": accuracy,
        "n": n,
        "accuracy_by_agreement": accuracy_by_agreement,
        "mean_kendall_tau": mean_tau,
        "n_prompts": len(taus),
    }


def compute_per_dimension_metrics(results: list[dict]) -> dict[str, dict]:
    by_dim: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_dim[r.get("dimension", "(unknown)")].append(r)
    return {dim: compute_eval_metrics(rs) for dim, rs in sorted(by_dim.items())}


# ---------------------------------------------------------------------------
# Hallucination metrics
# ---------------------------------------------------------------------------
def _binary_metrics(probs: list[float], labels: list[int]) -> dict:
    n = len(labels)
    if n == 0:
        return {"n": 0, "accuracy": 0.0, "f1": 0.0, "auc": 0.0,
                "tp": 0, "fp": 0, "fn": 0, "tn": 0, "positive_rate": 0.0}

    preds = [int(p >= 0.5) for p in probs]
    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
    tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)
    acc = (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    pos_scores = [s for s, y in zip(probs, labels) if y == 1]
    neg_scores = [s for s, y in zip(probs, labels) if y == 0]
    if pos_scores and neg_scores:
        u = 0.0
        for ps in pos_scores:
            for ns in neg_scores:
                if ps > ns:
                    u += 1
                elif ps == ns:
                    u += 0.5
        auc = u / (len(pos_scores) * len(neg_scores))
    else:
        auc = 0.0

    return {
        "n": n, "accuracy": acc, "f1": f1, "auc": auc,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "positive_rate": sum(labels) / n,
    }


def compute_halluc_metrics(rows: list[dict]) -> dict[str, dict]:
    """Per-template (and per-template×bucket) halluc metrics + 'overall' key."""
    if not rows:
        return {}
    overall = _binary_metrics(
        [float(r["prob"]) for r in rows], [int(r["label"]) for r in rows]
    )
    out: dict[str, dict] = {"overall": overall}
    by_template: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_template[r.get("prompt_template", "(none)")].append(r)
    for tmpl, rs in sorted(by_template.items()):
        m = _binary_metrics([float(r["prob"]) for r in rs], [int(r["label"]) for r in rs])
        by_bucket: dict[str, list[dict]] = defaultdict(list)
        for r in rs:
            by_bucket[r.get("agreement_bucket", "unanimous")].append(r)
        m["by_bucket"] = {
            label: _binary_metrics(
                [float(r["prob"]) for r in bs], [int(r["label"]) for r in bs]
            )
            for label, bs in sorted(by_bucket.items())
        }
        out[tmpl] = m
    return out


def print_halluc_metrics(metrics: dict, indent: str = "  ") -> None:
    if not metrics:
        return
    o = metrics["overall"]
    print(
        f"{indent}Halluc | acc={o['accuracy']:.1%}  f1={o['f1']:.3f}  "
        f"auc={o['auc']:.3f}  (n={o['n']}, pos={o['positive_rate']:.1%})"
    )
    for tmpl, m in metrics.items():
        if tmpl == "overall":
            continue
        print(
            f"{indent}  [{tmpl:<5}] acc={m['accuracy']:.1%}  f1={m['f1']:.3f}  "
            f"auc={m['auc']:.3f}  (n={m['n']}, pos={m['positive_rate']:.1%})"
        )
        for bucket, b in m.get("by_bucket", {}).items():
            if b["n"] == 0:
                continue
            print(
                f"{indent}    {bucket:>10}  acc={b['accuracy']:.1%}  "
                f"f1={b['f1']:.3f}  (n={b['n']})"
            )


def print_eval_metrics(metrics: dict, indent: str = "  ") -> None:
    print(f"{indent}Accuracy: {metrics['accuracy']:.1%}  (n={metrics.get('n', 0)})")
    for label in ("unanimous", "majority", "split"):
        acc, cnt = metrics["accuracy_by_agreement"][label]
        if cnt > 0:
            print(f"{indent}  {label:>10}: {acc:.1%}  ({cnt} pairs)")
    print(
        f"{indent}Kendall's tau (model vs human): "
        f"{metrics['mean_kendall_tau']:.3f}  ({metrics['n_prompts']} prompts)"
    )


def print_full_report(
    results: list[dict],
    halluc_results: list[dict] | None = None,
) -> None:
    overall = compute_eval_metrics(results)
    print("\n=== Overall ===")
    print_eval_metrics(overall)

    per_dim = compute_per_dimension_metrics(results)
    if len(per_dim) > 1:
        print("\n=== Per dimension ===")
        for dim, m in per_dim.items():
            print(f"\n[{dim}]")
            print_eval_metrics(m)

    if halluc_results:
        print("\n=== Hallucinations ===")
        print_halluc_metrics(compute_halluc_metrics(halluc_results))


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------
def _agreement_badge(r: dict) -> str:
    agr = float(r.get("agreement", 1.0))
    wr = r.get("win_rate_a", "")
    if agr >= 0.95:
        color, label = "#22c55e", "unanimous"
    elif agr >= 0.7:
        color, label = "#f59e0b", "majority"
    else:
        color, label = "#ef4444", "split"
    wr_str = f" (win_rate_a={wr})" if wr != "" else ""
    return (
        f' <span style="background:{color};color:#fff;padding:2px 6px;'
        f'border-radius:4px;font-size:11px">{label}{wr_str}</span>'
    )


def _dimension_badge(dim: str | None) -> str:
    if not dim:
        return ""
    return (
        f' <span style="background:#0ea5e9;color:#fff;padding:2px 6px;'
        f'border-radius:4px;font-size:11px;font-family:monospace">{html.escape(dim)}</span>'
    )


def _render_battle(r: dict) -> str:
    prompt_short = r["prompt"][:200] + ("..." if len(r["prompt"]) > 200 else "")
    winner_label = "A" if r["winner"] == "A" else "B"

    border_a = "3px solid #22c55e" if r["winner"] == "A" else "1px solid #ddd"
    border_b = "3px solid #22c55e" if r["winner"] == "B" else "1px solid #ddd"

    agr = float(r.get("agreement", 1.0))
    card_bg = "#fff" if agr >= 0.7 else "#fff7ed"

    pred_tag_a = pred_tag_b = ""
    pred_html = (
        ' <span style="background:#3b82f6;color:#fff;padding:2px 6px;'
        'border-radius:4px;font-size:12px">predicted</span>'
    )
    if r["predicted_winner"] == "A":
        pred_tag_a = pred_html
    else:
        pred_tag_b = pred_html

    status_color = "#22c55e" if r["correct"] else "#ef4444"
    status_text = "CORRECT" if r["correct"] else "WRONG"

    img_a_src = html.escape(r.get("image_url_a") or r["image_a"])
    img_b_src = html.escape(r.get("image_url_b") or r["image_b"])

    badge = _agreement_badge(r)
    dim_badge = _dimension_badge(r.get("dimension"))

    return f"""
    <div style="border:1px solid #e5e7eb;border-radius:8px;padding:16px;margin-bottom:16px;background:{card_bg}">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
        <span style="font-weight:600;color:{status_color}">{status_text}{badge}{dim_badge}</span>
        <span style="font-size:13px;color:#666">
          margin: {r['margin']:.4f} &nbsp;|&nbsp;
          ground truth winner: {winner_label} &nbsp;|&nbsp;
          {r.get('model_a','')} vs {r.get('model_b','')}
        </span>
      </div>
      <p style="font-size:13px;color:#374151;margin:0 0 12px 0;line-height:1.4">{html.escape(prompt_short)}</p>
      <div style="display:flex;gap:16px">
        <div style="flex:1;text-align:center">
          <img src="{img_a_src}" style="max-width:100%;max-height:300px;border-radius:6px;border:{border_a}" loading="lazy">
          <div style="margin-top:6px;font-size:13px">
            <b>A</b> ({r.get('model_a','')}) &nbsp; score: {r['score_a']:.4f}{pred_tag_a}
          </div>
        </div>
        <div style="flex:1;text-align:center">
          <img src="{img_b_src}" style="max-width:100%;max-height:300px;border-radius:6px;border:{border_b}" loading="lazy">
          <div style="margin-top:6px;font-size:13px">
            <b>B</b> ({r.get('model_b','')}) &nbsp; score: {r['score_b']:.4f}{pred_tag_b}
          </div>
        </div>
      </div>
    </div>"""


def _render_metrics_table(metrics: dict, title: str) -> str:
    rows = ""
    for label in ("unanimous", "majority", "split"):
        acc, cnt = metrics["accuracy_by_agreement"][label]
        if cnt > 0:
            rows += (
                f'<tr><td style="padding:4px 12px">{label}</td>'
                f'<td style="padding:4px 12px">{acc:.1%}</td>'
                f'<td style="padding:4px 12px">{cnt}</td></tr>\n'
            )
    return f"""
<div style="background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:16px;margin-bottom:24px">
  <h3 style="margin:0 0 8px 0">{html.escape(title)}</h3>
  <p style="margin:0 0 8px 0;font-size:14px;color:#374151">
    Accuracy: <b>{metrics['accuracy']:.1%}</b> (n={metrics['n']}) &nbsp;|&nbsp;
    Kendall's τ: <b>{metrics['mean_kendall_tau']:.3f}</b> ({metrics['n_prompts']} prompts)
  </p>
  <table style="border-collapse:collapse;font-size:14px">
    <tr style="border-bottom:1px solid #e5e7eb;font-weight:600">
      <td style="padding:4px 12px">Agreement</td><td style="padding:4px 12px">Accuracy</td><td style="padding:4px 12px">Pairs</td>
    </tr>
    {rows}
  </table>
</div>"""


def _render_halluc_metrics_html(halluc_metrics: dict) -> str:
    if not halluc_metrics:
        return ""
    o = halluc_metrics["overall"]
    sections = [f"""
<div style="background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:16px;margin-bottom:16px">
  <h3 style="margin:0 0 8px 0">Hallucination — overall</h3>
  <p style="margin:0;font-size:14px;color:#374151">
    Accuracy: <b>{o['accuracy']:.1%}</b> &nbsp;|&nbsp;
    F1: <b>{o['f1']:.3f}</b> &nbsp;|&nbsp;
    AUC: <b>{o['auc']:.3f}</b> &nbsp;|&nbsp;
    n={o['n']} (positives {o['positive_rate']:.1%})
  </p>
</div>"""]
    for tmpl, m in halluc_metrics.items():
        if tmpl == "overall":
            continue
        rows = ""
        for label in ("unanimous", "majority", "split"):
            b = m.get("by_bucket", {}).get(label)
            if not b or b["n"] == 0:
                continue
            rows += (
                f'<tr><td style="padding:4px 12px">{label}</td>'
                f'<td style="padding:4px 12px">{b["accuracy"]:.1%}</td>'
                f'<td style="padding:4px 12px">{b["f1"]:.3f}</td>'
                f'<td style="padding:4px 12px">{b["n"]}</td></tr>\n'
            )
        sections.append(f"""
<div style="background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:16px;margin-bottom:16px">
  <h3 style="margin:0 0 8px 0">Hallucination — {html.escape(tmpl)}</h3>
  <p style="margin:0 0 8px 0;font-size:14px;color:#374151">
    Accuracy: <b>{m['accuracy']:.1%}</b> &nbsp;|&nbsp;
    F1: <b>{m['f1']:.3f}</b> &nbsp;|&nbsp;
    AUC: <b>{m['auc']:.3f}</b> &nbsp;|&nbsp;
    n={m['n']} (positives {m['positive_rate']:.1%})
  </p>
  <table style="border-collapse:collapse;font-size:14px">
    <tr style="border-bottom:1px solid #e5e7eb;font-weight:600">
      <td style="padding:4px 12px">Agreement</td>
      <td style="padding:4px 12px">Accuracy</td>
      <td style="padding:4px 12px">F1</td>
      <td style="padding:4px 12px">n</td>
    </tr>
    {rows}
  </table>
</div>""")
    return "<h2 style=\"margin-top:32px\">Hallucination metrics</h2>\n" + "\n".join(sections)


def generate_html_report(
    results: list[dict],
    output_path: str,
    halluc_results: list[dict] | None = None,
) -> None:
    overall = compute_eval_metrics(results)
    per_dim = compute_per_dimension_metrics(results)
    correct = [r for r in results if r["correct"]]
    wrong = [r for r in results if not r["correct"]]

    wrong.sort(key=lambda r: r["margin"])
    correct.sort(key=lambda r: r["margin"])

    metrics_html = _render_metrics_table(overall, "Overall metrics")
    if len(per_dim) > 1:
        for dim, m in per_dim.items():
            metrics_html += _render_metrics_table(m, f"[{dim}]")

    halluc_metrics = (
        compute_halluc_metrics(halluc_results) if halluc_results else {}
    )
    metrics_html += _render_halluc_metrics_html(halluc_metrics)

    battles_html = ""
    if wrong:
        battles_html += '<h2 style="color:#ef4444;margin-top:32px">Failures</h2>\n'
        battles_html += (
            f'<p style="color:#666">{len(wrong)} battles &mdash; '
            f'sorted by margin (hardest first)</p>\n'
        )
        for r in wrong:
            battles_html += _render_battle(r)
    if correct:
        battles_html += '<h2 style="color:#22c55e;margin-top:32px">Successes</h2>\n'
        battles_html += (
            f'<p style="color:#666">{len(correct)} battles &mdash; '
            f'sorted by margin (least confident first)</p>\n'
        )
        for r in correct:
            battles_html += _render_battle(r)

    page = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>TASTE Eval Report</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;max-width:1024px;margin:0 auto;padding:24px;background:#f9fafb}}</style>
</head>
<body>
<h1>TASTE Evaluation Report</h1>
<div style="display:flex;gap:24px;margin-bottom:24px;flex-wrap:wrap">
  <div style="background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:16px;flex:1;text-align:center;min-width:120px">
    <div style="font-size:36px;font-weight:700">{overall['accuracy']:.1%}</div>
    <div style="color:#666">Accuracy</div>
  </div>
  <div style="background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:16px;flex:1;text-align:center;min-width:120px">
    <div style="font-size:36px;font-weight:700">{len(results)}</div>
    <div style="color:#666">Total battles</div>
  </div>
  <div style="background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:16px;flex:1;text-align:center;min-width:120px">
    <div style="font-size:36px;font-weight:700;color:#22c55e">{len(correct)}</div>
    <div style="color:#666">Correct</div>
  </div>
  <div style="background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:16px;flex:1;text-align:center;min-width:120px">
    <div style="font-size:36px;font-weight:700;color:#ef4444">{len(wrong)}</div>
    <div style="color:#666">Wrong</div>
  </div>
</div>
{metrics_html}
{battles_html}
</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(page)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _assert_images_exist(
    rows: list[dict],
    image_keys: tuple[str, ...],
    label: str,
    image_dir: Path | None,
    max_show: int = 10,
) -> None:
    """Verify every referenced image file exists; raise loudly otherwise.

    Mirrors ``train._assert_images_exist`` so a missing file at eval time
    fails fast with a clear message rather than producing degenerate
    embeddings via the VLM pre-processor's silent fallback.
    """
    missing: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in image_keys:
            p = r.get(k)
            if not p or p in seen:
                continue
            seen.add(p)
            if not os.path.exists(p):
                missing.append(p)
    if not missing:
        return
    sample = "\n  ".join(missing[:max_show])
    more = (
        f"\n  ... and {len(missing) - max_show} more"
        if len(missing) > max_show else ""
    )
    raise FileNotFoundError(
        f"{len(missing)} image file(s) referenced by {label} are missing on "
        f"disk under {image_dir}.\n"
        f"  First {min(max_show, len(missing))}:\n  {sample}{more}\n"
        f"Hint: confirm --image-dir matches where images actually live, or "
        f"rerun preprocess_data.py without --skip-download to fetch them."
    )


def cmd_score(args):
    scorer = HPSContraScorer(args.checkpoint, device=args.device)
    s = scorer.score(args.prompt, args.image, dimension=args.dimension)
    print(f"Score: {s:.4f}")


def cmd_compare(args):
    scorer = HPSContraScorer(args.checkpoint, device=args.device)
    result = scorer.compare(
        args.prompt, args.image_a, args.image_b, dimension=args.dimension
    )
    print(f"Score A: {result['score_a']:.4f}")
    print(f"Score B: {result['score_b']:.4f}")
    print(f"Winner:  {result['predicted_winner']} (margin: {result['margin']:.4f})")


def cmd_halluc(args):
    scorer = HPSContraScorer(args.checkpoint, device=args.device)
    r = scorer.hallucination(args.prompt, args.image)
    print(f"Hallucination probability: {r['prob']:.4f}")
    print(f"Predicted label: {r['predicted_label']}  (1 = hallucinates)")


def cmd_batch(args):
    scorer = HPSContraScorer(args.checkpoint, device=args.device)
    image_dir = Path(args.image_dir).resolve() if args.image_dir else None

    with open(args.input) as f:
        rows = list(csv.DictReader(f))

    def resolve(name: str) -> str:
        if image_dir and not os.path.isabs(name):
            return str(image_dir / name)
        return name

    results = []
    for row in tqdm(rows, desc="Scoring"):
        dim = row.get("dimension") or args.dimension
        if "image_a" in row and "image_b" in row:
            r = scorer.compare(
                row["prompt"], resolve(row["image_a"]), resolve(row["image_b"]),
                dimension=dim,
            )
            results.append({**row, **r})
        elif "image" in row:
            s = scorer.score(row["prompt"], resolve(row["image"]), dimension=dim)
            results.append({**row, "score": s})

    output = args.output or args.input.replace(".csv", "_scored.csv")
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {len(results)} results to {output}")


def cmd_eval(args):
    if not args.checkpoint and not args.baseline:
        raise SystemExit("Error: provide either --checkpoint or --baseline")
    if args.baseline:
        scorer = HPSContraScorer(baseline_model=args.baseline, device=args.device)
    else:
        scorer = HPSContraScorer(checkpoint_dir=args.checkpoint, device=args.device)

    image_dir = Path(args.image_dir).resolve() if args.image_dir else None
    if image_dir is not None and not image_dir.is_dir():
        raise FileNotFoundError(
            f"--image-dir does not exist or is not a directory: {image_dir}"
        )

    with open(args.val_csv, "r") as f:
        val_battles_raw = list(csv.DictReader(f))

    for b in val_battles_raw:
        if image_dir and not os.path.isabs(b["image_a"]):
            b["image_a"] = str(image_dir / b["image_a"])
            b["image_b"] = str(image_dir / b["image_b"])

    _assert_images_exist(
        val_battles_raw,
        image_keys=("image_a", "image_b"),
        label="ranking battles",
        image_dir=image_dir,
    )

    # Per-evaluator val files have up to 5 rows per pair; collapse to one
    # row per (dim, template, prompt_id, image_pair) so accuracy / Kendall
    # are computed over distinct pairs and not inflated 5×.
    seen: set[tuple] = set()
    val_battles: list[dict] = []
    for b in val_battles_raw:
        img_lo, img_hi = sorted([Path(b["image_a"]).name, Path(b["image_b"]).name])
        key = (b.get("dimension", ""), b.get("prompt_template", ""),
               b.get("prompt_id", ""), img_lo, img_hi)
        if key in seen:
            continue
        seen.add(key)
        val_battles.append(b)
    if len(val_battles) != len(val_battles_raw):
        print(
            f"  Val deduped {len(val_battles_raw)} → {len(val_battles)} "
            f"(per-evaluator → per-pair)"
        )

    print(f"Evaluating on {len(val_battles)} val battles ...")

    results = []
    for b in tqdm(val_battles, desc="Evaluating"):
        dim = b.get("dimension") or args.dimension
        r = scorer.compare(b["prompt"], b["image_a"], b["image_b"], dimension=dim)
        # Compare to the aggregated majority winner derived from win_rate_a
        # rather than the single evaluator's vote on the kept row.
        wr_a = float(b.get("win_rate_a") or (1.0 if b.get("winner") == "A" else 0.0))
        majority_winner = "A" if wr_a > 0.5 else "B"
        results.append({
            **b,
            "score_a": r["score_a"],
            "score_b": r["score_b"],
            "predicted_winner": r["predicted_winner"],
            "margin": r["margin"],
            "winner": majority_winner,
            "correct": r["predicted_winner"] == majority_winner,
        })

    halluc_results: list[dict] = []
    if args.halluc_val_csv:
        if scorer.halluc_head is None:
            print(
                "Warning: --halluc-val-csv given but checkpoint has no "
                "hallucination head; skipping halluc eval."
            )
        else:
            with open(args.halluc_val_csv) as f:
                halluc_rows = list(csv.DictReader(f))
            for r in halluc_rows:
                if image_dir and not os.path.isabs(r["image"]):
                    r["image"] = str(image_dir / r["image"])
            _assert_images_exist(
                halluc_rows,
                image_keys=("image",),
                label="hallucination assets",
                image_dir=image_dir,
            )
            print(f"Evaluating hallucination head on {len(halluc_rows)} assets ...")
            for r in tqdm(halluc_rows, desc="Halluc"):
                pred = scorer.hallucination(r["prompt"], r["image"])
                halluc_results.append({
                    **r,
                    "prob": pred["prob"],
                    "predicted_label": pred["predicted_label"],
                    "label": int(r["label"]),
                    "correct": pred["predicted_label"] == int(r["label"]),
                })

    print_full_report(results, halluc_results=halluc_results or None)

    output_html = args.output or "eval_report.html"
    generate_html_report(
        results, output_html,
        halluc_results=halluc_results or None,
    )
    print(f"\nReport saved to {output_html}")


def main():
    parser = argparse.ArgumentParser(description="TASTE inference")
    parser.add_argument("--device", type=str, default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    p_score = sub.add_parser("score")
    p_score.add_argument("--checkpoint", required=True)
    p_score.add_argument("--prompt", required=True)
    p_score.add_argument("--image", required=True)
    p_score.add_argument("--dimension", default=None,
                         help="Dimension slug (required if checkpoint has multiple).")

    p_compare = sub.add_parser("compare")
    p_compare.add_argument("--checkpoint", required=True)
    p_compare.add_argument("--prompt", required=True)
    p_compare.add_argument("--image-a", required=True)
    p_compare.add_argument("--image-b", required=True)
    p_compare.add_argument("--dimension", default=None)

    p_halluc = sub.add_parser("halluc", help="Predict hallucination probability for one image.")
    p_halluc.add_argument("--checkpoint", required=True)
    p_halluc.add_argument("--prompt", required=True)
    p_halluc.add_argument("--image", required=True)

    p_batch = sub.add_parser("batch")
    p_batch.add_argument("--checkpoint", required=True)
    p_batch.add_argument("--input", required=True)
    p_batch.add_argument("--image-dir", default=None)
    p_batch.add_argument("--output", default=None)
    p_batch.add_argument("--dimension", default=None,
                         help="Default dimension if rows don't carry one.")

    p_eval = sub.add_parser("eval", help="Run on val set and generate HTML report")
    p_eval.add_argument("--checkpoint", default=None,
                        help="Path to trained checkpoint (omit if using --baseline)")
    p_eval.add_argument("--baseline", default=None, metavar="MODEL",
                        help="Evaluate raw VLM with cosine similarity (e.g. openai/clip-vit-large-patch14)")
    p_eval.add_argument("--val-csv", required=True,
                        help="Path to battles_val.csv from preprocessing")
    p_eval.add_argument("--halluc-val-csv", default=None,
                        help="Path to halluc_val.csv. If checkpoint has a halluc head, "
                             "metrics + report sections are added.")
    p_eval.add_argument("--image-dir", default=None,
                        help="Directory containing images (for relative paths in CSV)")
    p_eval.add_argument("--dimension", default=None,
                        help="Default dimension when val rows lack one (legacy CSVs).")
    p_eval.add_argument("--output", default=None, help="Output HTML path")

    args = parser.parse_args()
    {
        "score": cmd_score,
        "compare": cmd_compare,
        "halluc": cmd_halluc,
        "batch": cmd_batch,
        "eval": cmd_eval,
    }[args.command](args)


if __name__ == "__main__":
    main()
