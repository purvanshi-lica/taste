"""
Public ``PreferenceScorer`` API for the TASTE preference model.

Loads a checkpoint produced by the upstream ``train.py --pairwise-head``
run (heads.pt + halluc_head.pt + meta.json) and scores image pairs along
each ranking dimension the head was trained on.

The scorer assumes the **pairwise-difference head** architecture: each
per-dimension MLP consumes the fused
``[t, i_a, i_b, i_a - i_b, |i_a - i_b|, t \\odot (i_a - i_b)]`` vector and
emits a Bradley-Terry logit directly.  The training-time temperature
``exp(logit_scale)`` is applied before sigmoid so the output is a
calibrated probability that ``image_a`` is preferred over ``image_b``.

The hallucination head (binary, per-image) is loaded if present and used
to emit a per-image hallucination probability.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from tqdm import tqdm

from taste_scorer.embedders import VLEmbedder
from taste_scorer.heads import HallucinationHead, PairwiseMultiHeadScorer


# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = ("prompt", "image_a", "image_b")
OPTIONAL_PASSTHROUGH_COLUMNS = ("pair_id",)


def _validate_dataframe(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input CSV is missing required column(s): {missing}.  "
            f"Required schema: {list(REQUIRED_COLUMNS)} "
            f"(plus optional {list(OPTIONAL_PASSTHROUGH_COLUMNS)})."
        )
    if df.empty:
        raise ValueError("Input CSV is empty.")
    # Check basic column types early so we fail fast on malformed input.
    for col in REQUIRED_COLUMNS:
        if df[col].isna().any():
            n = int(df[col].isna().sum())
            raise ValueError(f"Column '{col}' has {n} NA / empty value(s).")


def _resolve_image_paths(df: pd.DataFrame, image_dir: Path | None) -> pd.DataFrame:
    """Return df with image_a / image_b replaced by absolute paths.

    Paths in the CSV may be absolute (used as-is), relative (resolved
    against ``image_dir`` if given, otherwise against the current working
    directory), or filenames (always resolved against ``image_dir``).
    Raises ``FileNotFoundError`` for any path that does not exist on disk.
    """
    df = df.copy()
    for col in ("image_a", "image_b"):
        resolved: list[str] = []
        for raw in df[col]:
            p = Path(str(raw))
            if not p.is_absolute() and image_dir is not None:
                p = image_dir / p
            p = p.resolve()
            if not p.is_file():
                raise FileNotFoundError(
                    f"Image not found for column {col}: {p} "
                    f"(raw value: {raw!r})"
                )
            resolved.append(str(p))
        df[col] = resolved
    return df


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------
@dataclass
class _CheckpointMeta:
    model_name: str
    pairwise_head: bool
    has_halluc_head: bool
    dimensions: list[str]
    logit_scale: float  # raw parameter value; temperature = exp(logit_scale)
    enable_lora: bool
    head_input_layernorm: bool

    @classmethod
    def from_json(cls, path: Path) -> "_CheckpointMeta":
        with open(path) as f:
            data = json.load(f)
        return cls(
            model_name=data["model_name"],
            pairwise_head=bool(data.get("pairwise_head", False)),
            has_halluc_head=bool(data.get("has_halluc_head", False)),
            dimensions=list(data.get("dimensions", [])),
            # Older checkpoints sometimes omit logit_scale; the training default
            # initial value is log(10.0) ≈ 2.30, which we use as a fallback.
            logit_scale=float(data.get("logit_scale", math.log(10.0))),
            enable_lora=bool(data.get("enable_lora", False)),
            head_input_layernorm=bool(data.get("head_input_layernorm", True)),
        )


class PreferenceScorer:
    """Load a trained TASTE checkpoint and score image pairs.

    Public API:

        scorer = PreferenceScorer.from_checkpoint("path/to/best")
        df_out = scorer.score_pairs(df_in, image_dir="path/to/images/")

    where ``df_in`` has columns ``prompt``, ``image_a``, ``image_b`` (and
    optionally ``pair_id``), and ``df_out`` adds:

        prob_a_wins_<dim>   for each dimension the head was trained on
        halluc_prob_a       (only if the checkpoint has a halluc head)
        halluc_prob_b
    """

    def __init__(
        self,
        embedder: VLEmbedder,
        scorer: PairwiseMultiHeadScorer,
        halluc_head: HallucinationHead | None,
        logit_scale_value: float,
        device: torch.device,
    ):
        self.embedder = embedder
        self.scorer = scorer
        self.halluc_head = halluc_head
        self.device = device
        # We keep logit_scale as a plain float; it's the trained value and
        # never updated at inference time.
        self._temperature = math.exp(logit_scale_value)

    @property
    def dimensions(self) -> list[str]:
        return list(self.scorer.dimensions)

    @property
    def has_halluc_head(self) -> bool:
        return self.halluc_head is not None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        device: str | torch.device | None = None,
    ) -> "PreferenceScorer":
        ckpt_dir = Path(checkpoint_dir)
        if not ckpt_dir.is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

        meta = _CheckpointMeta.from_json(ckpt_dir / "meta.json")
        if not meta.pairwise_head:
            raise NotImplementedError(
                f"Checkpoint at {ckpt_dir} was trained without "
                f"--pairwise-head.  This package supports only the "
                f"pairwise-difference head architecture; retrain with "
                f"`bash retrain_best.sh` or use the scalar-path inference "
                f"code in the upstream repository."
            )

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        embedder = VLEmbedder.from_pretrained(meta.model_name, device=device)

        if meta.enable_lora:
            adapter_dir = ckpt_dir / "lora_adapter"
            if not adapter_dir.is_dir():
                raise FileNotFoundError(
                    f"meta.json says enable_lora=true but "
                    f"{adapter_dir} does not exist."
                )
            from peft import PeftModel

            embedder.model = PeftModel.from_pretrained(
                embedder.model, str(adapter_dir)
            )
        embedder.model.eval()

        heads_payload = torch.load(
            ckpt_dir / "heads.pt", map_location=device, weights_only=True
        )
        scorer = (
            PairwiseMultiHeadScorer.from_checkpoint(heads_payload).to(device).eval()
        )

        halluc_head: HallucinationHead | None = None
        if meta.has_halluc_head and (ckpt_dir / "halluc_head.pt").is_file():
            halluc_payload = torch.load(
                ckpt_dir / "halluc_head.pt", map_location=device, weights_only=True
            )
            halluc_head = (
                HallucinationHead.from_checkpoint(halluc_payload).to(device).eval()
            )

        return cls(
            embedder=embedder,
            scorer=scorer,
            halluc_head=halluc_head,
            logit_scale_value=meta.logit_scale,
            device=device,
        )

    # ------------------------------------------------------------------
    # Embedding helpers — dedup so we only encode each unique input once
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _encode_texts(self, texts: Iterable[str]) -> dict[str, torch.Tensor]:
        unique = sorted(set(texts))
        out: dict[str, torch.Tensor] = {}
        for t in tqdm(unique, desc="encode text", leave=False):
            out[t] = self.embedder.encode_text([t]).squeeze(0).to(torch.float32)
        return out

    @torch.no_grad()
    def _encode_images(self, paths: Iterable[str]) -> dict[str, torch.Tensor]:
        unique = sorted(set(paths))
        out: dict[str, torch.Tensor] = {}
        for p in tqdm(unique, desc="encode image", leave=False):
            out[p] = self.embedder.encode_image([p]).squeeze(0).to(torch.float32)
        return out

    # ------------------------------------------------------------------
    # Public scoring
    # ------------------------------------------------------------------
    @torch.no_grad()
    def score_pairs(
        self,
        df: pd.DataFrame,
        image_dir: str | Path | None = None,
        batch_size: int = 64,
    ) -> pd.DataFrame:
        """Score every (prompt, image_a, image_b) row in ``df``.

        Args:
            df: input DataFrame with columns ``prompt``, ``image_a``,
                ``image_b``; optional ``pair_id`` is passed through.
            image_dir: root directory used to resolve relative image paths
                in the CSV (absolute paths in the CSV override this).
            batch_size: number of pairs encoded into BT logits per
                forward pass.  Image / text embedding is per-unique-value
                (one forward per unique path / prompt), so this only
                affects the head-forward step.

        Returns:
            A new DataFrame with the input columns plus
            ``prob_a_wins_<dim>`` for each dimension, and
            ``halluc_prob_a`` / ``halluc_prob_b`` if a halluc head is
            present.
        """
        _validate_dataframe(df)
        image_dir_path = Path(image_dir).resolve() if image_dir else None
        df_resolved = _resolve_image_paths(df, image_dir_path)

        text_emb = self._encode_texts(df_resolved["prompt"].tolist())
        img_emb = self._encode_images(
            list(df_resolved["image_a"]) + list(df_resolved["image_b"])
        )

        # Stack into tensors aligned with df_resolved row order.
        def _stack(col_or_values: Iterable[str], lookup: dict[str, torch.Tensor]) -> torch.Tensor:
            return torch.stack([lookup[v] for v in col_or_values]).to(self.device)

        t_all = _stack(df_resolved["prompt"], text_emb)
        a_all = _stack(df_resolved["image_a"], img_emb)
        b_all = _stack(df_resolved["image_b"], img_emb)

        n_rows = len(df_resolved)

        # Pairwise scores per dimension.
        probs: dict[str, list[float]] = {dim: [] for dim in self.dimensions}
        for dim in self.dimensions:
            for start in range(0, n_rows, batch_size):
                end = min(start + batch_size, n_rows)
                logit = self.scorer.pair_logit(
                    t_all[start:end], a_all[start:end], b_all[start:end], dim
                )
                scaled = logit.to(torch.float32) * self._temperature
                probs[dim].extend(torch.sigmoid(scaled).cpu().tolist())

        # Halluc scores per image (one forward per side).
        halluc_a: list[float] | None = None
        halluc_b: list[float] | None = None
        if self.halluc_head is not None:
            def _halluc(text_t: torch.Tensor, img_t: torch.Tensor) -> list[float]:
                out: list[float] = []
                for start in range(0, n_rows, batch_size):
                    end = min(start + batch_size, n_rows)
                    logit = self.halluc_head(
                        text_t[start:end], img_t[start:end]
                    )
                    out.extend(torch.sigmoid(logit).to(torch.float32).cpu().tolist())
                return out

            halluc_a = _halluc(t_all, a_all)
            halluc_b = _halluc(t_all, b_all)

        # Assemble the output DataFrame.
        passthrough_cols = [
            c for c in OPTIONAL_PASSTHROUGH_COLUMNS if c in df.columns
        ] + list(REQUIRED_COLUMNS)
        # Use the original (un-resolved) image paths in the output for
        # round-trip readability; the resolved ones were only needed for
        # the model.
        out_df = df[passthrough_cols].copy().reset_index(drop=True)
        for dim in self.dimensions:
            out_df[f"prob_a_wins_{dim}"] = probs[dim]
        if halluc_a is not None and halluc_b is not None:
            out_df["halluc_prob_a"] = halluc_a
            out_df["halluc_prob_b"] = halluc_b
        return out_df
