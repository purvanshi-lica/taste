"""
Train TASTE ranking + hallucination heads.

Architecture (current default):

    text, image ──VLM (frozen)──▶ text_emb, image_emb
                                       │
                                       ├─▶ MLP_d  (per-dimension BT score)
                                       └─▶ MLP_halluc (binary BCE)

There is one MLP head per evaluation dimension (``preference``,
``typography``, ``color_harmony`` …) plus one shared hallucination head
that takes a single (text, image) pair and emits a logit.  The VLM
backbone is shared and frozen by default; pass ``--enable-lora`` to
additionally fit a LoRA adapter on top of the encoder.

Losses
------
* **Bradley-Terry pairwise** for the dimension heads, with a CLIP-style
  learned temperature ``s = exp(logit_scale)`` and per-pair
  ``agreement`` weighting (ambiguous pairs contribute proportionally
  less; we never drop them).
* **BCE-with-logits on a soft label** for hallucinations, weighted by
  the asset's cross-evaluator agreement.

The per-step total loss is ``L = L_bt + λ · L_halluc`` where ``λ`` is
``--halluc-loss-weight``; we draw one BT mini-batch and one halluc
mini-batch each step (cycled independently of the BT epoch length).

Validation reports per-dimension BT metrics (accuracy + per-bucket
breakdown + Kendall τ) and per-prompt-template hallucination metrics
(accuracy, F1, AUC) — both jointly and per-bucket.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import itertools
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BitsAndBytesConfig

try:
    import wandb
except ImportError:
    wandb = None

from embed_cache import EmbeddingCache
from embedders import VLEmbedder
from heads import HallucinationHead, MultiHeadScorer, PairwiseMultiHeadScorer
from inference import (
    compute_eval_metrics,
    compute_per_dimension_metrics,
    generate_html_report,
    print_eval_metrics,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen3-VL-Embedding-2B"
    train_csv: str = "data/battles_train.csv"
    val_csv: str = "data/battles_val.csv"
    halluc_train_csv: str | None = "data/halluc_train.csv"
    halluc_val_csv: str | None = "data/halluc_val.csv"
    image_dir: str = "data/images"
    output_dir: str = "checkpoints"

    # Heads
    head_hidden_dim: int = 512
    head_dropout: float = 0.1
    head_input_layernorm: bool = True
    # Pairwise head: emits the BT logit from (text, img_a, img_b) directly
    # instead of two independent scalar scores.  See heads.PairwiseScoringMLP.
    pairwise_head: bool = False

    # Tier 1 — criterion-conditioned text prompts.  When True, every battle
    # row has a per-dimension prefix prepended to its prompt before encoding,
    # so the text embedding becomes criterion-aware.  Halluc rows use a
    # generic "Detect hallucinations." prefix.
    criterion_conditioned_prompt: bool = False

    # LoRA (opt-in)
    enable_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.15
    lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj")

    # Backbone
    load_in_4bit: bool = False
    torch_dtype: str = "bfloat16"
    low_vram: bool = False

    # Optimisation
    lr_heads: float = 1e-3
    lr_lora: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 30
    batch_size: int = 32
    halluc_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.05
    patience: int = 5
    seed: int = 42

    # Loss
    agreement_weighting: bool = True
    halluc_loss_weight: float = 1.0
    soft_labels: bool = False  # if True, use aggregated win_rate_a instead of per-row hard label
    initial_logit_scale: float = 10.0  # exp of initial logit_scale parameter
    log_every: int = 50  # emit one wandb training point per N optimizer steps

    # Caching
    cache_embeddings: bool = True  # in-memory only, rebuilt every run; auto-disabled when LoRA is on

    resume: bool = False
    wandb_project: str | None = None
    wandb_run_name: str | None = None  # auto-generated from key params when None


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
# Per-dimension prefix used when ``criterion_conditioned_prompt`` is on.
# We bake the criterion into the prompt text (rather than into the embedder's
# instruction) so the embedding cache naturally separates entries by
# (criterion, prompt) without any cache-key changes.
PROMPT_PREFIX_BY_DIM: dict[str, str] = {
    "preference":          "Focus on overall preference quality. ",
    "typography":          "Focus on typography quality. ",
    "color_harmony":       "Focus on color harmony. ",
    "mood_and_color_tone": "Focus on mood and color tone. ",
    "visual_hierarchy":    "Focus on visual hierarchy. ",
    "color_accuracy":      "Focus on color accuracy. ",
    "spatial_accuracy":    "Focus on spatial accuracy. ",
}
HALLUC_PROMPT_PREFIX = "Detect hallucinations and rendering errors. "


def _apply_criterion_prefix(prompt: str, dimension: str) -> str:
    """Prepend the per-dimension prefix to ``prompt``.  Unknown dims pass through."""
    prefix = PROMPT_PREFIX_BY_DIM.get(dimension)
    if not prefix:
        return prompt
    return prefix + prompt


class BattleDataset(Dataset):
    def __init__(self, battles: list[dict], criterion_conditioned: bool = False):
        self.battles = battles
        self.criterion_conditioned = criterion_conditioned

    def __len__(self) -> int:
        return len(self.battles)

    def __getitem__(self, idx: int) -> dict:
        b = self.battles[idx]
        if b.get("win_rate_a", "") != "":
            win_rate_a = float(b["win_rate_a"])
        else:
            win_rate_a = 1.0 if b["winner"] == "A" else 0.0
        # ``target`` is the per-row hard label (this evaluator's vote when the
        # row is per-evaluator; the majority winner when the row is aggregated).
        # ``win_rate_a`` is the soft label aggregated across all 5 evaluators
        # of the same (dim, template, prompt_id, image_pair); it's identical
        # for the 5 per-evaluator rows of a single pair.
        target = 1.0 if b["winner"] == "A" else 0.0
        prompt = b["prompt"]
        if self.criterion_conditioned:
            prompt = _apply_criterion_prefix(prompt, b["dimension"])
        return {
            "prompt": prompt,
            "image_a": b["image_a"],
            "image_b": b["image_b"],
            "dimension": b["dimension"],
            "prompt_template": b.get("prompt_template", ""),
            "target": target,
            "win_rate_a": win_rate_a,
            "agreement": float(b.get("agreement", 1.0)),
            "agreement_bucket": b.get("agreement_bucket", "unanimous"),
            "winner": b["winner"],
            "prompt_id": b.get("prompt_id", ""),
            "image_url_a": b.get("image_url_a", ""),
            "image_url_b": b.get("image_url_b", ""),
            "model_a": b.get("model_a", ""),
            "model_b": b.get("model_b", ""),
            "rank_a": float(b.get("mean_rank_a", b.get("rank_a", 0))),
            "rank_b": float(b.get("mean_rank_b", b.get("rank_b", 0))),
        }


def battle_collate(batch: list[dict]) -> dict:
    out = {k: [item[k] for item in batch] for k in batch[0].keys()}
    out["target_t"] = torch.tensor(out["target"], dtype=torch.float32)
    out["win_rate_a_t"] = torch.tensor(out["win_rate_a"], dtype=torch.float32)
    out["agreement_t"] = torch.tensor(out["agreement"], dtype=torch.float32)
    return out


def _format_lr(lr: float) -> str:
    """Compact human-readable formatting for learning-rate values."""
    if lr <= 0:
        return "0"
    exp = math.floor(math.log10(lr))
    mant = lr / (10 ** exp)
    if abs(mant - round(mant)) < 1e-6:
        return f"{int(round(mant))}e{int(exp)}"
    return f"{mant:.1f}e{int(exp)}"


def make_run_name(cfg: "TrainConfig") -> str:
    """Build a descriptive wandb run name from the most-tuned hyper-parameters.

    Convention (in order, separated by ``-``):

      backbone-mode | head-shape | regularisation | optimisation | data-policy

    Example: ``mlp-h128-d0.3-wd0.05-lr1e-3-bs32-cos-aw1-hardlabels``.

    Tweaks that are often varied (capacity, dropout, weight-decay, LR, batch
    size, LoRA presence) appear early; less-varied switches (label policy,
    agreement weighting) come at the end.
    """
    parts: list[str] = []

    # Backbone mode -------------------------------------------------------
    if cfg.enable_lora:
        parts.append(f"lora-r{cfg.lora_r}a{cfg.lora_alpha}")
    else:
        parts.append("mlp")

    # Head shape ----------------------------------------------------------
    parts.append(f"h{cfg.head_hidden_dim}")
    parts.append(f"d{cfg.head_dropout:g}")

    # Regularisation ------------------------------------------------------
    parts.append(f"wd{cfg.weight_decay:g}")

    # Optimisation --------------------------------------------------------
    if cfg.enable_lora:
        parts.append(f"lrh{_format_lr(cfg.lr_heads)}")
        parts.append(f"lrl{_format_lr(cfg.lr_lora)}")
    else:
        parts.append(f"lr{_format_lr(cfg.lr_heads)}")
    parts.append(f"bs{cfg.batch_size}")
    parts.append(f"ls{int(cfg.initial_logit_scale)}")

    # Data / loss policy --------------------------------------------------
    parts.append("soft" if cfg.soft_labels else "hard")
    if not cfg.agreement_weighting:
        parts.append("noaw")

    # Tier 1 flags --------------------------------------------------------
    if cfg.pairwise_head:
        parts.append("pair")
    if cfg.criterion_conditioned_prompt:
        parts.append("ccp")
    if not cfg.head_input_layernorm:
        parts.append("noln")

    return "-".join(parts)


def _assert_images_exist(
    rows: list[dict],
    image_keys: tuple[str, ...],
    label: str,
    image_dir: Path,
    max_show: int = 10,
) -> None:
    """Verify every referenced image file is on disk; raise loudly otherwise.

    Without this, a single missing file silently propagated all the way
    through the VLM pre-processor (see ``model.prepare_inputs``) and ended
    up degenerating contrastive training to ``log(2)``.  Failing fast at
    load time is much friendlier than debugging that downstream.
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


def dedupe_battles_for_val(battles: list[dict]) -> list[dict]:
    """Collapse per-evaluator val rows to one row per (dim, template, pid, pair).

    Each pair appears up to 5 times in the per-evaluator val CSV; per-pair
    metrics (accuracy, Kendall) are well-defined only on the deduped form.
    """
    seen: set[tuple] = set()
    out: list[dict] = []
    for b in battles:
        img_lo, img_hi = sorted([Path(b["image_a"]).name, Path(b["image_b"]).name])
        key = (
            b.get("dimension", ""),
            b.get("prompt_template", ""),
            b.get("prompt_id", ""),
            img_lo, img_hi,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(b)
    return out


class HallucinationDataset(Dataset):
    def __init__(self, rows: list[dict], criterion_conditioned: bool = False):
        self.rows = rows
        self.criterion_conditioned = criterion_conditioned

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        r = self.rows[idx]
        prompt = r["prompt"]
        if self.criterion_conditioned:
            prompt = HALLUC_PROMPT_PREFIX + prompt
        return {
            "prompt": prompt,
            "image": r["image"],
            "halluc_rate": float(r["halluc_rate"]),
            "label": int(r["label"]),
            "agreement": float(r.get("agreement", 1.0)),
            "agreement_bucket": r.get("agreement_bucket", "unanimous"),
            "prompt_template": r.get("prompt_template", ""),
            "prompt_id": r.get("prompt_id", ""),
            "model": r.get("model", ""),
            "asset_id": r.get("asset_id", ""),
            "image_url": r.get("image_url", ""),
        }


def halluc_collate(batch: list[dict]) -> dict:
    out = {k: [item[k] for item in batch] for k in batch[0].keys()}
    out["halluc_rate_t"] = torch.tensor(out["halluc_rate"], dtype=torch.float32)
    out["agreement_t"] = torch.tensor(out["agreement"], dtype=torch.float32)
    out["label_t"] = torch.tensor(out["label"], dtype=torch.long)
    return out


def load_battles(csv_path: str) -> list[dict]:
    with open(csv_path, "r") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Forward pass — embedding cache or live VLM
# ---------------------------------------------------------------------------
def encode_text_image_pair(
    embedder: VLEmbedder,
    cache: EmbeddingCache | None,
    prompts: list[str],
    images: list[str],
    device: torch.device,
    autocast_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (text_emb, image_emb) for a list of (prompt, image) pairs."""
    if cache is not None:
        return (
            cache.text(prompts, device=device, dtype=torch.float32),
            cache.image(images, device=device, dtype=torch.float32),
        )

    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=autocast_dtype)
        if autocast_dtype is not None and device.type == "cuda"
        else torch.autocast(device_type="cpu", enabled=False)
    )
    with autocast_ctx:
        text_emb = embedder.encode_text(prompts).to(torch.float32)
        image_emb = embedder.encode_image(images).to(torch.float32)
    return text_emb, image_emb


def encode_battle_batch(
    embedder: VLEmbedder,
    cache: EmbeddingCache | None,
    batch: dict,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    text_emb, img_a_emb = encode_text_image_pair(
        embedder, cache, batch["prompt"], batch["image_a"], device, autocast_dtype
    )
    # Reuse cached text; only image_b is new.
    if cache is not None:
        img_b_emb = cache.image(batch["image_b"], device=device, dtype=torch.float32)
    else:
        autocast_ctx = (
            torch.autocast(device_type=device.type, dtype=autocast_dtype)
            if autocast_dtype is not None and device.type == "cuda"
            else torch.autocast(device_type="cpu", enabled=False)
        )
        with autocast_ctx:
            img_b_emb = embedder.encode_image(batch["image_b"]).to(torch.float32)
    return text_emb, img_a_emb, img_b_emb


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def bt_loss(
    score_a: torch.Tensor,
    score_b: torch.Tensor,
    win_rate_a: torch.Tensor,
    logit_scale: torch.nn.Parameter,
    agreement: torch.Tensor | None = None,
) -> torch.Tensor:
    scale = logit_scale.exp().clamp(max=100.0).to(torch.float32)
    diff = (score_a - score_b).to(torch.float32) * scale
    return _bt_loss_from_logit(diff, win_rate_a, agreement)


def bt_loss_from_pair_logit(
    pair_logit: torch.Tensor,
    win_rate_a: torch.Tensor,
    logit_scale: torch.nn.Parameter,
    agreement: torch.Tensor | None = None,
) -> torch.Tensor:
    """BT loss for the pairwise-head path.

    ``pair_logit`` is the head's direct output (already representing
    ``score_a − score_b`` in some learned scale); we still multiply by
    ``exp(logit_scale)`` so the temperature parameter has the same
    semantics as in the scalar path.
    """
    scale = logit_scale.exp().clamp(max=100.0).to(torch.float32)
    diff = pair_logit.to(torch.float32) * scale
    return _bt_loss_from_logit(diff, win_rate_a, agreement)


def _bt_loss_from_logit(
    diff: torch.Tensor,
    win_rate_a: torch.Tensor,
    agreement: torch.Tensor | None,
) -> torch.Tensor:
    p = win_rate_a.to(torch.float32)
    per_example = -(p * F.logsigmoid(diff) + (1 - p) * F.logsigmoid(-diff))
    if agreement is not None:
        w = agreement.to(torch.float32)
        denom = w.sum().clamp(min=1e-6)
        return (per_example * w).sum() / denom
    return per_example.mean()


def _print_first_batch_diagnostics(
    text_emb: torch.Tensor,
    img_a_emb: torch.Tensor,
    img_b_emb: torch.Tensor,
    score_a: torch.Tensor,
    score_b: torch.Tensor,
    logit_scale: torch.nn.Parameter,
    target: torch.Tensor,
) -> None:
    """One-shot diagnostic on the first training batch.

    Looks for the failure modes that produce a flat ``log(2)`` loss:

    1. *Degenerate embeddings* — cache returning identical vectors for every
       prompt or image (the model literally can't distinguish samples).
    2. *Saturated init* — L2-normalised embeddings → tiny fused features →
       MLP outputs ~0 for everything, BCE stuck at log(2).
    3. *score_a ≡ score_b* — wiring bug somewhere between cache and loss.
    """
    def _stats(name: str, t: torch.Tensor) -> str:
        norms = t.norm(dim=-1) if t.ndim > 1 else t.abs()
        return (
            f"  {name:<14} shape={tuple(t.shape)}  "
            f"|x|.mean={t.abs().mean().item():.4f}  "
            f"row_norm.mean={norms.mean().item():.4f}  "
            f"row_std={t.std().item():.4f}"
        )

    def _pairwise_distinctness(name: str, t: torch.Tensor) -> str:
        # Mean pairwise cosine similarity across rows. If all rows are
        # the same vector, this is ≈ 1.0 (and the cache is broken).
        if t.size(0) < 2:
            return f"  {name:<14} only 1 row, skipping pairwise check"
        n = t / (t.norm(dim=-1, keepdim=True) + 1e-12)
        cos = n @ n.t()
        n_rows = cos.size(0)
        off_diag = (cos.sum() - cos.diag().sum()) / (n_rows * (n_rows - 1))
        return (
            f"  {name:<14} mean off-diag cosine = {off_diag.item():+.4f}  "
            f"(≈1 ⇒ all rows identical; ≈0 ⇒ uncorrelated)"
        )

    diff = (score_a - score_b)
    scaled = diff * logit_scale.exp().to(diff.dtype)
    print("\n" + "─" * 72)
    print("First-batch diagnostics (run once at start of training):")
    print(_stats("text_emb",  text_emb))
    print(_stats("img_a_emb", img_a_emb))
    print(_stats("img_b_emb", img_b_emb))
    print(_stats("score_a",   score_a))
    print(_stats("score_b",   score_b))
    print(_stats("score_diff", diff))
    print(_stats("scaled_diff", scaled))

    print("  ── pairwise distinctness across batch rows ──")
    with torch.no_grad():
        print(_pairwise_distinctness("text_emb",  text_emb.float()))
        print(_pairwise_distinctness("img_a_emb", img_a_emb.float()))
        print(_pairwise_distinctness("img_b_emb", img_b_emb.float()))

        # img_a vs img_b on the *same* row: are the two images in each pair
        # actually different?  Cosine ≈ 1 ⇒ image_a == image_b for every row,
        # which would deterministically give score_a ≡ score_b and loss = log(2).
        a = img_a_emb.float() / (img_a_emb.float().norm(dim=-1, keepdim=True) + 1e-12)
        b = img_b_emb.float() / (img_b_emb.float().norm(dim=-1, keepdim=True) + 1e-12)
        per_row_cos = (a * b).sum(dim=-1)
        print(
            f"  img_a vs img_b within-row cosine: mean={per_row_cos.mean().item():+.4f}  "
            f"min={per_row_cos.min().item():+.4f}  max={per_row_cos.max().item():+.4f}  "
            f"(<1 ⇒ images in each pair are distinct)"
        )

    pred = (score_a > score_b).float()
    gt = (target > 0.5).float()
    print(
        f"  initial agreement with target: {(pred == gt).float().mean().item():.3f}  "
        f"(50% ⇒ uninformative)"
    )

    # Heuristic warnings ----------------------------------------------------
    warnings = []
    if scaled.abs().mean().item() < 1e-3:
        warnings.append(
            "scaled_diff ≈ 0: heads are at the saturated init. Confirm "
            "ScoringMLP.use_input_layernorm=True and try --initial-logit-scale 50."
        )
    if (score_a - score_b).abs().max().item() < 1e-6:
        warnings.append(
            "score_a ≡ score_b across the entire batch.  This means the MLP "
            "produces identical output for img_a and img_b — likely because "
            "img_a_emb == img_b_emb (broken cache) or a wiring bug in score_grouped."
        )
    if score_a.std().item() < 1e-6:
        warnings.append(
            "score_a has zero variance across rows.  All inputs to the MLP "
            "are mapping to the same scalar.  Likely degenerate embeddings."
        )
    for w in warnings:
        print(f"  ⚠ {w}")

    print("─" * 72)


def halluc_loss(
    logits: torch.Tensor,
    halluc_rate: torch.Tensor,
    agreement: torch.Tensor | None = None,
) -> torch.Tensor:
    """BCE-with-logits on the soft label ``halluc_rate`` ∈ [0, 1]."""
    targets = halluc_rate.to(torch.float32)
    per_example = F.binary_cross_entropy_with_logits(
        logits.to(torch.float32), targets, reduction="none"
    )
    if agreement is not None:
        w = agreement.to(torch.float32)
        denom = w.sum().clamp(min=1e-6)
        return (per_example * w).sum() / denom
    return per_example.mean()


# ---------------------------------------------------------------------------
# Hallucination metrics
# ---------------------------------------------------------------------------
def _binary_metrics(probs: list[float], labels: list[int]) -> dict:
    """Accuracy / F1 / AUC for a binary head, no scipy/sklearn dependency."""
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

    # AUC via Mann-Whitney U statistic on the score ordering.
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
        "n": n,
        "accuracy": acc,
        "f1": f1,
        "auc": auc,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "positive_rate": sum(labels) / n,
    }


def per_template_halluc_metrics(rows: list[dict]) -> dict[str, dict]:
    """Compute halluc metrics overall, per template, and per template×bucket."""
    overall = _binary_metrics([r["prob"] for r in rows], [r["label"] for r in rows])
    per_template: dict[str, dict] = {"overall": overall}
    by_template: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_template[r.get("prompt_template", "(none)")].append(r)
    for tmpl, rs in sorted(by_template.items()):
        m = _binary_metrics([r["prob"] for r in rs], [r["label"] for r in rs])
        per_template[tmpl] = m
        # Per-bucket breakdown.
        by_bucket: dict[str, list[dict]] = defaultdict(list)
        for r in rs:
            by_bucket[r.get("agreement_bucket", "unanimous")].append(r)
        per_template[tmpl]["by_bucket"] = {
            label: _binary_metrics([r["prob"] for r in bs], [r["label"] for r in bs])
            for label, bs in sorted(by_bucket.items())
        }
    return per_template


def print_halluc_metrics(metrics: dict, indent: str = "  ") -> None:
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
                f"{indent}    {bucket:>10} acc={b['accuracy']:.1%}  "
                f"f1={b['f1']:.3f}  (n={b['n']})"
            )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_battle_validation(
    embedder: VLEmbedder,
    cache: EmbeddingCache | None,
    scorer: "MultiHeadScorer | PairwiseMultiHeadScorer",
    logit_scale: torch.nn.Parameter,
    val_loader: DataLoader,
    val_battles: list[dict],
    device: torch.device,
    autocast_dtype: torch.dtype | None,
) -> tuple[float, list[dict]]:
    scorer.eval()
    pairwise = isinstance(scorer, PairwiseMultiHeadScorer)

    results: list[dict] = []
    total_loss = 0.0
    total_n = 0
    cursor = 0

    for batch in tqdm(val_loader, desc="val[BT]"):
        text_emb, img_a_emb, img_b_emb = encode_battle_batch(
            embedder, cache, batch, device, autocast_dtype
        )
        if pairwise:
            pair_logit = scorer.pair_logit_grouped(
                text_emb, img_a_emb, img_b_emb, batch["dimension"]
            )
            score_a = 0.5 * pair_logit
            score_b = -0.5 * pair_logit
        else:
            score_a = scorer.score_grouped(text_emb, img_a_emb, batch["dimension"])
            score_b = scorer.score_grouped(text_emb, img_b_emb, batch["dimension"])
        # Validate on the soft win_rate_a so split (3-2) pairs contribute
        # the right partial loss instead of being graded on a single
        # evaluator's vote.
        wr = batch["win_rate_a_t"].to(device)
        if pairwise:
            loss = bt_loss_from_pair_logit(pair_logit, wr, logit_scale, agreement=None)
        else:
            loss = bt_loss(score_a, score_b, wr, logit_scale, agreement=None)

        bs = wr.size(0)
        total_loss += loss.item() * bs
        total_n += bs
        sa_list = score_a.detach().cpu().tolist()
        sb_list = score_b.detach().cpu().tolist()
        for i in range(bs):
            b = val_battles[cursor + i]
            sa, sb = sa_list[i], sb_list[i]
            pred = "A" if sa > sb else "B"
            # Compare to the aggregated majority winner (derived from
            # win_rate_a) rather than the single evaluator's vote on the
            # row we happened to keep during dedupe.  win_rate_a is on the
            # 0.0 / 0.2 / 0.4 / 0.6 / 0.8 / 1.0 grid (5 evaluators), never
            # 0.5, so the majority side is always well-defined.
            wr_a = float(b.get("win_rate_a", 1.0 if b["winner"] == "A" else 0.0))
            majority_winner = "A" if wr_a > 0.5 else "B"
            results.append({
                **b,
                "score_a": sa,
                "score_b": sb,
                "predicted_winner": pred,
                "margin": abs(sa - sb),
                "winner": majority_winner,
                "correct": pred == majority_winner,
            })
        cursor += bs
    return total_loss / max(1, total_n), results


@torch.no_grad()
def run_halluc_validation(
    embedder: VLEmbedder,
    cache: EmbeddingCache | None,
    halluc_head: HallucinationHead,
    val_loader: DataLoader,
    val_rows: list[dict],
    device: torch.device,
    autocast_dtype: torch.dtype | None,
) -> tuple[float, list[dict]]:
    halluc_head.eval()
    total_loss = 0.0
    total_n = 0
    cursor = 0
    results: list[dict] = []

    for batch in tqdm(val_loader, desc="val[halluc]"):
        text_emb, img_emb = encode_text_image_pair(
            embedder, cache, batch["prompt"], batch["image"], device, autocast_dtype
        )
        logits = halluc_head(text_emb, img_emb)
        rate = batch["halluc_rate_t"].to(device)
        loss = halluc_loss(logits, rate, agreement=None)
        bs = rate.size(0)
        total_loss += loss.item() * bs
        total_n += bs

        probs = torch.sigmoid(logits).detach().cpu().tolist()
        for i in range(bs):
            r = val_rows[cursor + i]
            results.append({
                **r,
                "prob": probs[i],
                "predicted_label": int(probs[i] >= 0.5),
                "label": int(r["label"]),
                "correct": int(probs[i] >= 0.5) == int(r["label"]),
            })
        cursor += bs
    return total_loss / max(1, total_n), results


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(cfg: TrainConfig):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, cfg.torch_dtype)
    print(f"Device: {device}, dtype: {dtype}")

    # ── Load VLM backbone ──────────────────────────────────────────────
    quant_config = None
    if cfg.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    embedder_kwargs: dict = {}
    if cfg.low_vram:
        from model import MAX_PIXELS_LOW
        embedder_kwargs["max_pixels"] = MAX_PIXELS_LOW

    embedder = VLEmbedder.from_pretrained(
        cfg.model_name, device=device, torch_dtype=dtype,
        quantization_config=quant_config, **embedder_kwargs,
    )
    backbone = embedder.model

    # ── Load battles + halluc data ────────────────────────────────────
    image_dir = Path(cfg.image_dir).resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"--image-dir does not exist or is not a directory: {image_dir}"
        )

    def resolve_battle(rows: list[dict]) -> list[dict]:
        for b in rows:
            b["image_a"] = str(image_dir / b["image_a"])
            b["image_b"] = str(image_dir / b["image_b"])
        return rows

    def resolve_halluc(rows: list[dict]) -> list[dict]:
        for r in rows:
            r["image"] = str(image_dir / r["image"])
        return rows

    train_battles = resolve_battle(load_battles(cfg.train_csv))
    val_battles_raw = resolve_battle(load_battles(cfg.val_csv))
    if not train_battles:
        raise RuntimeError(f"No training battles found in {cfg.train_csv}")

    _assert_images_exist(
        train_battles + val_battles_raw,
        image_keys=("image_a", "image_b"),
        label="ranking battles",
        image_dir=image_dir,
    )

    # Val rows may be per-evaluator (5 rows per pair); collapse to one row per
    # pair so accuracy / Kendall are computed over distinct pairs rather than
    # being trivially inflated 5×.
    val_battles = dedupe_battles_for_val(val_battles_raw)
    if len(val_battles) != len(val_battles_raw):
        print(
            f"  Val deduped {len(val_battles_raw)} → {len(val_battles)} "
            f"(one row per (dim, template, prompt_id, image_pair))"
        )

    dimensions = sorted({b["dimension"] for b in train_battles + val_battles})
    print(
        f"Loaded {len(train_battles)} train + {len(val_battles)} val battles "
        f"across {len(dimensions)} dimensions: {dimensions}"
    )

    # Quick agreement-bucket histogram on training set so the user can
    # eyeball signal quality before kicking off long jobs.
    bucket_train: dict[str, int] = defaultdict(int)
    for b in train_battles:
        bucket_train[b.get("agreement_bucket", "unanimous")] += 1
    print(
        f"  Train bucket histogram: "
        f"unanimous={bucket_train.get('unanimous', 0)}, "
        f"majority={bucket_train.get('majority', 0)}, "
        f"split={bucket_train.get('split', 0)}"
    )

    halluc_train: list[dict] = []
    halluc_val: list[dict] = []
    if cfg.halluc_train_csv and Path(cfg.halluc_train_csv).exists():
        halluc_train = resolve_halluc(load_battles(cfg.halluc_train_csv))
    if cfg.halluc_val_csv and Path(cfg.halluc_val_csv).exists():
        halluc_val = resolve_halluc(load_battles(cfg.halluc_val_csv))
    if halluc_train or halluc_val:
        _assert_images_exist(
            halluc_train + halluc_val,
            image_keys=("image",),
            label="hallucination assets",
            image_dir=image_dir,
        )
    if halluc_train:
        print(f"Loaded {len(halluc_train)} halluc train + {len(halluc_val)} val rows")

    # ── LoRA (optional) ────────────────────────────────────────────────
    if cfg.enable_lora:
        from peft import LoraConfig, TaskType, get_peft_model
        lora_cfg = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
            target_modules=list(cfg.lora_target_modules),
            lora_dropout=cfg.lora_dropout, bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        backbone = get_peft_model(backbone, lora_cfg)
        embedder.model = backbone
        for p in backbone.parameters():
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.to(torch.float32)
        if hasattr(backbone, "print_trainable_parameters"):
            backbone.print_trainable_parameters()
        if hasattr(backbone, "enable_input_require_grads"):
            backbone.enable_input_require_grads()
        if hasattr(backbone, "gradient_checkpointing_enable"):
            backbone.gradient_checkpointing_enable()
    else:
        for p in backbone.parameters():
            p.requires_grad = False
        backbone.eval()

    # ── Embedding cache (frozen-backbone fast path) ────────────────────
    cache: EmbeddingCache | None = None
    if cfg.cache_embeddings and not cfg.enable_lora:
        cache = EmbeddingCache(embedder)
        prompts = (
            [b["prompt"] for b in train_battles + val_battles]
            + [r["prompt"] for r in halluc_train + halluc_val]
        )
        images = (
            [b["image_a"] for b in train_battles + val_battles]
            + [b["image_b"] for b in train_battles + val_battles]
            + [r["image"] for r in halluc_train + halluc_val]
        )
        cache.precompute_texts(prompts)
        cache.precompute_images(images)
        emb_dim = cache.emb_dim
    else:
        with torch.no_grad():
            sample = embedder.encode_text([train_battles[0]["prompt"]])
            emb_dim = int(sample.shape[-1])

    print(f"Embedding dim = {emb_dim}")

    # ── Heads ──────────────────────────────────────────────────────────
    if cfg.pairwise_head:
        scorer = PairwiseMultiHeadScorer(
            dimensions=dimensions, emb_dim=emb_dim,
            hidden_dim=cfg.head_hidden_dim, dropout=cfg.head_dropout,
            use_input_layernorm=cfg.head_input_layernorm,
        ).to(device=device, dtype=torch.float32)
        n_head_params = sum(p.numel() for p in scorer.parameters())
        print(
            f"Pairwise BT heads: {len(dimensions)} × MLP({6*emb_dim} → {cfg.head_hidden_dim}) "
            f"| total params: {n_head_params:,}"
        )
    else:
        scorer = MultiHeadScorer(
            dimensions=dimensions, emb_dim=emb_dim,
            hidden_dim=cfg.head_hidden_dim, dropout=cfg.head_dropout,
            use_input_layernorm=cfg.head_input_layernorm,
        ).to(device=device, dtype=torch.float32)
        n_head_params = sum(p.numel() for p in scorer.parameters())
        print(
            f"BT heads: {len(dimensions)} × MLP({4*emb_dim} → {cfg.head_hidden_dim}) "
            f"| total params: {n_head_params:,}"
        )

    halluc_head: HallucinationHead | None = None
    if halluc_train:
        halluc_head = HallucinationHead(
            emb_dim=emb_dim, hidden_dim=cfg.head_hidden_dim, dropout=cfg.head_dropout,
            use_input_layernorm=cfg.head_input_layernorm,
        ).to(device=device, dtype=torch.float32)
        print(f"Halluc head: 1 × MLP({4*emb_dim} → {cfg.head_hidden_dim} → 1)")

    # ── Logit scale ───────────────────────────────────────────────────
    logit_scale = torch.nn.Parameter(
        torch.tensor(math.log(cfg.initial_logit_scale), dtype=torch.float32, device=device)
    )
    print(f"Initial logit_scale = {logit_scale.exp().item():.2f}")

    # ── Loaders ────────────────────────────────────────────────────────
    train_loader = DataLoader(
        BattleDataset(train_battles, criterion_conditioned=cfg.criterion_conditioned_prompt),
        batch_size=cfg.batch_size, shuffle=True,
        collate_fn=battle_collate, drop_last=True,
    )
    val_loader = DataLoader(
        BattleDataset(val_battles, criterion_conditioned=cfg.criterion_conditioned_prompt),
        batch_size=cfg.batch_size, shuffle=False,
        collate_fn=battle_collate,
    )
    halluc_loader = halluc_val_loader = None
    if halluc_train:
        halluc_loader = DataLoader(
            HallucinationDataset(halluc_train, criterion_conditioned=cfg.criterion_conditioned_prompt),
            batch_size=cfg.halluc_batch_size,
            shuffle=True, collate_fn=halluc_collate, drop_last=True,
        )
        halluc_val_loader = DataLoader(
            HallucinationDataset(halluc_val, criterion_conditioned=cfg.criterion_conditioned_prompt),
            batch_size=cfg.halluc_batch_size,
            shuffle=False, collate_fn=halluc_collate,
        )

    # ── W&B ────────────────────────────────────────────────────────────
    use_wandb = cfg.wandb_project is not None and wandb is not None
    if cfg.wandb_project and wandb is None:
        print("Warning: --wandb-project set but wandb not installed, skipping")
    if use_wandb:
        run_name = cfg.wandb_run_name or make_run_name(cfg)
        print(f"W&B run name: {run_name}")
        wandb.init(
            project=cfg.wandb_project,
            name=run_name,
            config=dataclasses.asdict(cfg),
        )
        wandb.watch(scorer, log="gradients", log_freq=50)
        if halluc_head is not None:
            wandb.watch(halluc_head, log="gradients", log_freq=50)

    # ── Optimiser ─────────────────────────────────────────────────────
    head_params = list(scorer.parameters())
    if halluc_head is not None:
        head_params += list(halluc_head.parameters())
    param_groups = [
        {"params": head_params, "lr": cfg.lr_heads},
        {"params": [logit_scale], "weight_decay": 0.0, "lr": cfg.lr_heads},
    ]
    if cfg.enable_lora:
        lora_params = [p for p in backbone.parameters() if p.requires_grad]
        if lora_params:
            param_groups.insert(0, {"params": lora_params, "lr": cfg.lr_lora})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)

    accum = cfg.gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(train_loader) / accum)
    total_steps = max(1, steps_per_epoch * cfg.epochs)
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Resume ─────────────────────────────────────────────────────────
    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    start_epoch = 0
    history: list[dict] = []

    last_dir = output_dir / "last"
    if cfg.resume and (last_dir / "meta.json").exists():
        with open(last_dir / "meta.json") as f:
            last_meta = json.load(f)
        start_epoch = last_meta["epoch"]
        best_val_acc = last_meta.get("val_acc", 0.0)
        print(f"Resuming from epoch {start_epoch} (val_acc={best_val_acc:.3f})")

        if (last_dir / "heads.pt").exists():
            scorer.load_state_dict(torch.load(
                last_dir / "heads.pt", map_location=device, weights_only=True
            )["state_dict"])
        if halluc_head is not None and (last_dir / "halluc_head.pt").exists():
            halluc_head.load_state_dict(torch.load(
                last_dir / "halluc_head.pt", map_location=device, weights_only=True
            )["state_dict"])
        if cfg.enable_lora and (last_dir / "lora_adapter").exists():
            backbone.load_adapter(str(last_dir / "lora_adapter"), adapter_name="default")
        state_path = last_dir / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location=device, weights_only=True)
            optimizer.load_state_dict(state["optimizer"])
            if state.get("scheduler") is not None:
                scheduler.load_state_dict(state["scheduler"])
            if "logit_scale" in state:
                with torch.no_grad():
                    logit_scale.copy_(state["logit_scale"].to(
                        device=device, dtype=torch.float32
                    ))

        history_path = output_dir / "history.json"
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)

    # ── Training loop ─────────────────────────────────────────────────
    autocast_dtype = dtype if cfg.enable_lora else None  # cache path is fp32
    epochs_without_improvement = 0
    diagnostics_done = False

    # Step-level logging window.  Per-batch wandb points are too noisy at
    # 286 it/epoch, so we average over `cfg.log_every` optimizer steps and
    # emit one point per window keyed on `global_step`.
    global_step = 0
    log_window: dict[str, list[float]] = defaultdict(list)

    def _flush_window(extra: dict[str, float] | None = None) -> None:
        if not log_window or not use_wandb:
            log_window.clear()
            return
        payload = {k: float(np.mean(v)) for k, v in log_window.items()}
        payload["train/epoch"] = epoch + 1
        if extra:
            payload.update(extra)
        wandb.log(payload, step=global_step)
        log_window.clear()

    for epoch in range(start_epoch, cfg.epochs):
        scorer.train()
        if halluc_head is not None:
            halluc_head.train()
        if cfg.enable_lora:
            backbone.train()

        # The halluc loader is much smaller than the battles loader; cycle it
        # so we always have a halluc batch when we have a BT batch.
        halluc_iter = itertools.cycle(halluc_loader) if halluc_loader else None

        epoch_bt_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        epoch_halluc_loss = 0.0
        epoch_halluc_n = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [train]")
        for step_idx, batch in enumerate(pbar):
            text_emb, img_a_emb, img_b_emb = encode_battle_batch(
                embedder, cache, batch, device, autocast_dtype
            )
            if cfg.pairwise_head:
                pair_logit = scorer.pair_logit_grouped(
                    text_emb, img_a_emb, img_b_emb, batch["dimension"]
                )
                # For diagnostics / accuracy bookkeeping we synthesise
                # "scores" from the pair logit so the rest of the loop
                # (accuracy calc, _print_first_batch_diagnostics) is
                # unchanged.  ``score_a := pair_logit / 2`` and
                # ``score_b := −pair_logit / 2`` so ``score_a − score_b
                # = pair_logit`` exactly.
                score_a = 0.5 * pair_logit
                score_b = -0.5 * pair_logit
            else:
                score_a = scorer.score_grouped(text_emb, img_a_emb, batch["dimension"])
                score_b = scorer.score_grouped(text_emb, img_b_emb, batch["dimension"])
            if cfg.soft_labels:
                target = batch["win_rate_a_t"].to(device)
            else:
                target = batch["target_t"].to(device)
            agr = batch["agreement_t"].to(device) if cfg.agreement_weighting else None
            if cfg.pairwise_head:
                loss_bt = bt_loss_from_pair_logit(pair_logit, target, logit_scale, agreement=agr)
            else:
                loss_bt = bt_loss(score_a, score_b, target, logit_scale, agreement=agr)
            loss = loss_bt

            if not diagnostics_done:
                _print_first_batch_diagnostics(
                    text_emb, img_a_emb, img_b_emb,
                    score_a, score_b, logit_scale, target,
                )
                diagnostics_done = True

            loss_halluc_value: float | None = None
            if halluc_iter is not None and halluc_head is not None:
                hb = next(halluc_iter)
                ht_emb, hi_emb = encode_text_image_pair(
                    embedder, cache, hb["prompt"], hb["image"], device, autocast_dtype
                )
                hlogits = halluc_head(ht_emb, hi_emb)
                hagr = hb["agreement_t"].to(device) if cfg.agreement_weighting else None
                loss_h = halluc_loss(hlogits, hb["halluc_rate_t"].to(device), agreement=hagr)
                loss = loss + cfg.halluc_loss_weight * loss_h
                loss_halluc_value = loss_h.item()
                epoch_halluc_loss += loss_halluc_value * hb["halluc_rate_t"].size(0)
                epoch_halluc_n += hb["halluc_rate_t"].size(0)

            (loss / accum).backward()

            with torch.no_grad():
                pred_a = (score_a > score_b).float()
                gt_a = (target > 0.5).float()
                n_correct = (pred_a == gt_a).sum().item()
            bs = target.size(0)

            if (step_idx + 1) % accum == 0 or (step_idx + 1) == len(train_loader):
                params = list(scorer.parameters()) + [logit_scale]
                if halluc_head is not None:
                    params += list(halluc_head.parameters())
                if cfg.enable_lora:
                    params += [p for p in backbone.parameters() if p.requires_grad]
                grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                logit_scale_grad = (
                    logit_scale.grad.item() if logit_scale.grad is not None else 0.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                if use_wandb:
                    # Match the per-epoch summary's metric names so both
                    # streams populate the same wandb charts.
                    log_window["train/bt_loss"].append(loss_bt.item())
                    log_window["train/lr_heads"].append(scheduler.get_last_lr()[-1])
                    log_window["train/logit_scale"].append(logit_scale.exp().item())
                    log_window["train/grad_norm"].append(float(grad_norm))
                    log_window["train/logit_scale_grad"].append(logit_scale_grad)
                    if loss_halluc_value is not None:
                        log_window["train/halluc_loss"].append(loss_halluc_value)

                    if global_step % cfg.log_every == 0:
                        _flush_window()

            epoch_bt_loss += loss_bt.item() * bs
            epoch_correct += n_correct
            epoch_total += bs
            postfix = {
                "loss_bt": f"{loss_bt.item():.4f}",
                "acc": f"{epoch_correct/max(1, epoch_total):.3f}",
            }
            if loss_halluc_value is not None:
                postfix["loss_h"] = f"{loss_halluc_value:.4f}"
            pbar.set_postfix(**postfix)

        # Flush any partial step-window so the per-epoch summary point lands
        # at the same global_step as the last training samples.
        _flush_window()

        train_bt_loss = epoch_bt_loss / max(1, epoch_total)
        train_acc = epoch_correct / max(1, epoch_total)
        train_h_loss = (
            epoch_halluc_loss / max(1, epoch_halluc_n) if epoch_halluc_n else 0.0
        )

        # ── Validation ────────────────────────────────────────────────
        val_bt_loss, results = run_battle_validation(
            embedder, cache, scorer, logit_scale,
            val_loader, val_battles, device, autocast_dtype,
        )
        overall = compute_eval_metrics(results)
        per_dim = compute_per_dimension_metrics(results)
        val_acc = overall["accuracy"]

        halluc_metrics: dict | None = None
        halluc_results: list[dict] = []
        val_h_loss = 0.0
        if halluc_head is not None and halluc_val_loader is not None:
            val_h_loss, halluc_results = run_halluc_validation(
                embedder, cache, halluc_head, halluc_val_loader, halluc_val,
                device, autocast_dtype,
            )
            halluc_metrics = per_template_halluc_metrics(halluc_results)

        print(
            f"\nEpoch {epoch+1}: "
            f"train_loss_bt={train_bt_loss:.4f} train_acc={train_acc:.3f} "
            f"train_loss_h={train_h_loss:.4f} | "
            f"val_loss_bt={val_bt_loss:.4f} val_acc={val_acc:.3f} "
            f"val_loss_h={val_h_loss:.4f}"
        )
        print("Overall BT metrics:")
        print_eval_metrics(overall)
        for dim, m in per_dim.items():
            print(f"\n[{dim}]")
            print_eval_metrics(m)
        if halluc_metrics is not None:
            print("\nHallucination metrics:")
            print_halluc_metrics(halluc_metrics)

        # HTML reports
        reports_dir = output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        generate_html_report(results, str(reports_dir / f"eval_epoch_{epoch+1}.html"))
        for dim in per_dim:
            generate_html_report(
                [r for r in results if r["dimension"] == dim],
                str(reports_dir / f"eval_epoch_{epoch+1}__{dim}.html"),
            )

        history.append({
            "epoch": epoch + 1,
            "train_bt_loss": train_bt_loss,
            "train_acc": train_acc,
            "train_h_loss": train_h_loss,
            "val_bt_loss": val_bt_loss,
            "val_acc": val_acc,
            "val_h_loss": val_h_loss,
            "overall": overall,
            "per_dim": per_dim,
            "halluc": halluc_metrics,
        })
        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        if use_wandb:
            log = {
                "epoch": epoch + 1,
                "train/bt_loss": train_bt_loss,
                "train/halluc_loss": train_h_loss,
                "train/accuracy": train_acc,
                "val/bt_loss": val_bt_loss,
                "val/accuracy": val_acc,
                "val/halluc_loss": val_h_loss,
                "val/kendall_tau": overall["mean_kendall_tau"],
            }
            for label in ("unanimous", "majority", "split"):
                acc, cnt = overall["accuracy_by_agreement"][label]
                if cnt > 0:
                    log[f"val/accuracy_{label}"] = acc
            for dim, m in per_dim.items():
                log[f"val/{dim}/accuracy"] = m["accuracy"]
                log[f"val/{dim}/kendall_tau"] = m["mean_kendall_tau"]
                for label in ("unanimous", "majority", "split"):
                    acc, cnt = m["accuracy_by_agreement"][label]
                    if cnt > 0:
                        log[f"val/{dim}/accuracy_{label}"] = acc
            if halluc_metrics is not None:
                for tmpl, m in halluc_metrics.items():
                    log[f"val/halluc/{tmpl}/accuracy"] = m["accuracy"]
                    log[f"val/halluc/{tmpl}/f1"] = m["f1"]
                    log[f"val/halluc/{tmpl}/auc"] = m["auc"]
            wandb.log(log, step=global_step)

        save_checkpoint(
            scorer, halluc_head, backbone, cfg, output_dir / "last",
            epoch + 1, val_acc, logit_scale=logit_scale,
            optimizer=optimizer, scheduler=scheduler,
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            save_checkpoint(
                scorer, halluc_head, backbone, cfg, output_dir / "best",
                epoch + 1, val_acc, logit_scale=logit_scale,
            )
            print(f"  → New best val_acc: {val_acc:.3f}")
        else:
            epochs_without_improvement += 1
            if cfg.patience > 0 and epochs_without_improvement >= cfg.patience:
                print(
                    f"  Early stopping at epoch {epoch+1} "
                    f"(no improvement for {cfg.patience} epochs)"
                )
                break

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.3f}")
    print(f"Checkpoints saved to {output_dir}")

    if use_wandb:
        wandb.finish()


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
def save_checkpoint(
    scorer: "MultiHeadScorer | PairwiseMultiHeadScorer",
    halluc_head: HallucinationHead | None,
    backbone,
    cfg: TrainConfig,
    save_dir: Path,
    epoch: int,
    val_acc: float,
    logit_scale: torch.nn.Parameter | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
):
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(scorer.state_dict_for_checkpoint(), save_dir / "heads.pt")
    if halluc_head is not None:
        torch.save(halluc_head.state_dict_for_checkpoint(), save_dir / "halluc_head.pt")

    if cfg.enable_lora and hasattr(backbone, "save_pretrained"):
        backbone.save_pretrained(str(save_dir / "lora_adapter"))

    meta = {
        "epoch": epoch,
        "val_acc": val_acc,
        "model_name": cfg.model_name,
        "head_hidden_dim": cfg.head_hidden_dim,
        "head_dropout": cfg.head_dropout,
        "head_input_layernorm": cfg.head_input_layernorm,
        "pairwise_head": cfg.pairwise_head,
        "criterion_conditioned_prompt": cfg.criterion_conditioned_prompt,
        "enable_lora": cfg.enable_lora,
        "dimensions": scorer.dimensions,
        "has_halluc_head": halluc_head is not None,
    }
    if logit_scale is not None:
        meta["logit_scale"] = float(logit_scale.detach().cpu().item())
    with open(save_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if optimizer is not None:
        state = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
        }
        if logit_scale is not None:
            state["logit_scale"] = logit_scale.detach().cpu()
        torch.save(state, save_dir / "training_state.pt")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train multi-head TASTE")
    parser.add_argument("--train-csv", default="data/battles_train.csv")
    parser.add_argument("--val-csv", default="data/battles_val.csv")
    parser.add_argument("--halluc-train-csv", default="data/halluc_train.csv")
    parser.add_argument("--halluc-val-csv", default="data/halluc_val.csv")
    parser.add_argument(
        "--no-halluc", action="store_true",
        help="Disable hallucination head training even if halluc CSVs exist.",
    )
    parser.add_argument("--image-dir", default="data/images")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-Embedding-2B")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--halluc-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr-heads", type=float, default=1e-3)
    parser.add_argument("--lr-lora", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="AdamW weight_decay applied to non-bias head params.")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--head-hidden-dim", type=int, default=512)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--halluc-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--no-head-input-layernorm", action="store_true",
        help=(
            "Disable the input LayerNorm in ScoringMLP / HallucinationHead. "
            "The LayerNorm rescales the fused (L2-normalised) features so "
            "the MLP first layer doesn't saturate near zero; disabling it "
            "is a useful ablation."
        ),
    )
    parser.add_argument(
        "--pairwise-head", action="store_true",
        help=(
            "Use PairwiseMultiHeadScorer: each head sees (text, img_a, "
            "img_b) and emits the BT logit directly, instead of producing "
            "two independent scalar scores.  Tier 1 experiment for cracking "
            "the val-acc ceiling."
        ),
    )
    parser.add_argument(
        "--criterion-conditioned-prompt", action="store_true",
        help=(
            "Prepend a per-dimension prefix (e.g. 'Focus on typography "
            "quality. ') to every battle prompt before encoding, so the "
            "text embedding becomes criterion-aware.  Tier 1 experiment."
        ),
    )

    parser.add_argument(
        "--enable-lora", action="store_true",
        help="Train a LoRA adapter on the VLM backbone alongside the heads."
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.15)
    parser.add_argument(
        "--lora-target-modules", nargs="+", default=["q_proj", "v_proj"]
    )

    parser.add_argument(
        "--no-agreement-weighting", action="store_true",
        help="Disable per-pair agreement weighting (use unweighted losses)."
    )
    parser.add_argument(
        "--no-cache-embeddings", action="store_true",
        help=(
            "Disable the in-memory embedding cache and force a live VLM "
            "forward on every batch.  By default the cache is enabled: it "
            "lives only in-process (no disk persistence, no cross-run "
            "reuse) and is rebuilt from scratch at the start of every run, "
            "so there's no stale-cache risk between runs.  Auto-disabled "
            "anyway when --enable-lora is on."
        ),
    )
    parser.add_argument(
        "--soft-labels", action="store_true",
        help=(
            "Use the aggregated soft win_rate_a as the BT target instead of "
            "the per-row hard winner.  Recommended when the train CSV is "
            "already deduped to one row per pair (preprocess --dedupe). "
            "When the train CSV is per-evaluator (the default), hard labels "
            "give 5× more independent gradient signal per pair."
        ),
    )
    parser.add_argument(
        "--initial-logit-scale", type=float, default=10.0,
        help=(
            "Initial value of exp(logit_scale).  Larger values amplify the "
            "BT logit at init, which helps escape the saturated init when "
            "the backbone emits L2-normalised embeddings."
        ),
    )
    parser.add_argument(
        "--log-every", type=int, default=50,
        help=(
            "Emit one wandb training point per N optimizer steps (averaged "
            "over the window).  Default 50.  Per-epoch summary points are "
            "always emitted regardless of this setting."
        ),
    )

    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--low-vram", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument(
        "--wandb-run-name", default=None,
        help=(
            "Wandb run name.  If omitted, an auto-generated descriptive name "
            "based on the most-tuned hyper-parameters is used "
            "(e.g. mlp-h128-d0.3-wd0.05-lr1e-3-bs32-ls10-hard)."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.low_vram:
        args.load_in_4bit = True

    cfg = TrainConfig(
        model_name=args.model,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        halluc_train_csv=None if args.no_halluc else args.halluc_train_csv,
        halluc_val_csv=None if args.no_halluc else args.halluc_val_csv,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        head_hidden_dim=args.head_hidden_dim,
        head_dropout=args.head_dropout,
        head_input_layernorm=not args.no_head_input_layernorm,
        pairwise_head=args.pairwise_head,
        criterion_conditioned_prompt=args.criterion_conditioned_prompt,
        halluc_loss_weight=args.halluc_loss_weight,
        enable_lora=args.enable_lora,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=tuple(args.lora_target_modules),
        load_in_4bit=args.load_in_4bit,
        torch_dtype="float16" if args.low_vram else "bfloat16",
        low_vram=args.low_vram,
        lr_heads=args.lr_heads, lr_lora=args.lr_lora,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=1 if args.low_vram else args.batch_size,
        halluc_batch_size=1 if args.low_vram else args.halluc_batch_size,
        gradient_accumulation_steps=8 if args.low_vram else args.gradient_accumulation_steps,
        agreement_weighting=not args.no_agreement_weighting,
        cache_embeddings=not args.no_cache_embeddings,
        soft_labels=args.soft_labels,
        initial_logit_scale=args.initial_logit_scale,
        log_every=args.log_every,
        patience=args.patience,
        resume=args.resume,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        seed=args.seed,
    )
    train(cfg)


if __name__ == "__main__":
    main()
