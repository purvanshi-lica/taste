"""
Per-dimension MLP scoring heads + binary hallucination head for HPS-Contra.

Each evaluation dimension (e.g. ``preference``, ``typography``,
``color_harmony``…) gets its own small MLP that maps the *fused* (text,
image) embedding pair to a scalar quality score for that dimension.  All
heads share the same VLM backbone — and the same instruction-tuned text /
image encoders — so they benefit from the same general visual-language
priors while learning dimension-specific scoring rules.

A separate :class:`HallucinationHead` consumes a single (text, image) pair
and emits one binary logit (BCE-with-logits) for predicting whether the
image hallucinates content not described by the prompt.

Feature design.  We concatenate four cross-modal features that have proven
robust in NLI- and CLIP-style scoring heads:

* ``text``                — raw text embedding
* ``image``               — raw image embedding
* ``text * image``        — Hadamard product (alignment)
* ``|text - image|``      — absolute distance

This gives the MLP both the unimodal vectors and explicit interaction terms
without requiring it to learn them from scratch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def fuse_features(text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
    """Concatenate ``[text, image, text*image, |text-image|]`` along the feature dim."""
    return torch.cat(
        [text_emb, image_emb, text_emb * image_emb, (text_emb - image_emb).abs()],
        dim=-1,
    )


class ScoringMLP(nn.Module):
    """A 2-hidden-layer MLP that maps fused (text, image) features → scalar score.

    Used both as a per-dimension scoring head (BT pairwise loss) and, with its
    raw output reinterpreted as a logit, as the hallucination head (BCE).

    Important — input scaling.  Qwen3-VL-Embedding (and most modern embedding
    models) emit **L2-normalised** embeddings of dimensionality ``emb_dim``;
    each component is O(1/√emb_dim).  Without rescaling, the fused 4·emb_dim
    vector is tiny in magnitude and the MLP's first ``Linear(4·emb_dim → h)``
    produces near-zero outputs at default init, so the BT logit is ~0 and
    BCE loss is stuck at ``log(2)``.  We fix this with a ``LayerNorm`` on the
    fused input, which standardises features to unit variance per element
    independent of the upstream embedder's norm.
    """

    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        use_input_layernorm: bool = True,
    ):
        super().__init__()
        in_dim = 4 * emb_dim
        self.use_input_layernorm = use_input_layernorm
        self.input_norm = nn.LayerNorm(in_dim) if use_input_layernorm else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        x = fuse_features(text_emb, image_emb)
        x = self.input_norm(x)
        return self.net(x).squeeze(-1)


class HallucinationHead(nn.Module):
    """Single binary head predicting "any hallucination" from (text, image).

    The output is a raw logit; pass it to ``binary_cross_entropy_with_logits``
    against a soft label in ``[0, 1]`` (the per-asset evaluator-vote share),
    or to ``sigmoid`` for a probability at inference time.
    """

    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        use_input_layernorm: bool = True,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.use_input_layernorm = use_input_layernorm
        self.mlp = ScoringMLP(
            emb_dim, hidden_dim=hidden_dim, dropout=dropout,
            use_input_layernorm=use_input_layernorm,
        )

    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        return self.mlp(text_emb, image_emb)

    def state_dict_for_checkpoint(self) -> dict:
        return {
            "state_dict": self.state_dict(),
            "config": {
                "emb_dim": self.emb_dim,
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout,
                "use_input_layernorm": self.use_input_layernorm,
            },
        }

    @classmethod
    def from_checkpoint(cls, payload: dict) -> "HallucinationHead":
        cfg = payload["config"]
        m = cls(
            emb_dim=cfg["emb_dim"],
            hidden_dim=cfg.get("hidden_dim", 512),
            dropout=cfg.get("dropout", 0.1),
            # Older checkpoints predate this flag and have no LayerNorm weights.
            use_input_layernorm=cfg.get("use_input_layernorm", False),
        )
        m.load_state_dict(payload["state_dict"])
        return m


class MultiHeadScorer(nn.Module):
    """One :class:`ScoringMLP` per dimension, dispatched by string id.

    All heads share the same architecture but have independent parameters
    so each can specialise to its dimension's labelling rule.
    """

    def __init__(
        self,
        dimensions: list[str],
        emb_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        use_input_layernorm: bool = True,
    ):
        super().__init__()
        self.dimensions = sorted(dimensions)
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.use_input_layernorm = use_input_layernorm
        self.heads = nn.ModuleDict({
            dim: ScoringMLP(
                emb_dim, hidden_dim, dropout,
                use_input_layernorm=use_input_layernorm,
            )
            for dim in self.dimensions
        })

    def score(
        self,
        text_emb: torch.Tensor,
        image_emb: torch.Tensor,
        dimension: str,
    ) -> torch.Tensor:
        """Score a homogeneous batch (all examples share ``dimension``)."""
        if dimension not in self.heads:
            raise KeyError(
                f"Unknown dimension '{dimension}'. Known: {list(self.heads.keys())}"
            )
        return self.heads[dimension](text_emb, image_emb)

    def score_grouped(
        self,
        text_emb: torch.Tensor,
        image_emb: torch.Tensor,
        dimensions: list[str],
    ) -> torch.Tensor:
        """Score a heterogeneous batch: ``dimensions[i]`` selects the head for row *i*.

        We group rows by dimension to do one MLP forward per dimension, then
        scatter the results back into the original batch order.  This avoids
        a Python-level per-row loop while keeping the head dispatch correct.
        """
        if len(dimensions) != text_emb.size(0):
            raise ValueError("len(dimensions) must equal batch size")

        out = torch.empty(text_emb.size(0), device=text_emb.device, dtype=text_emb.dtype)
        unique = sorted(set(dimensions))
        for dim in unique:
            idx = [i for i, d in enumerate(dimensions) if d == dim]
            idx_t = torch.tensor(idx, device=text_emb.device, dtype=torch.long)
            sub_text = text_emb.index_select(0, idx_t)
            sub_img = image_emb.index_select(0, idx_t)
            scores = self.heads[dim](sub_text, sub_img)
            out.index_copy_(0, idx_t, scores.to(out.dtype))
        return out

    def state_dict_for_checkpoint(self) -> dict:
        """Return a serialisable state dict + architectural metadata."""
        return {
            "state_dict": self.state_dict(),
            "config": {
                "dimensions": self.dimensions,
                "emb_dim": self.emb_dim,
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout,
                "use_input_layernorm": self.use_input_layernorm,
            },
        }

    @classmethod
    def from_checkpoint(cls, payload: dict) -> "MultiHeadScorer":
        cfg = payload["config"]
        m = cls(
            dimensions=cfg["dimensions"],
            emb_dim=cfg["emb_dim"],
            hidden_dim=cfg.get("hidden_dim", 512),
            dropout=cfg.get("dropout", 0.1),
            # Older checkpoints predate this flag and have no LayerNorm weights.
            use_input_layernorm=cfg.get("use_input_layernorm", False),
        )
        m.load_state_dict(payload["state_dict"])
        return m


# ---------------------------------------------------------------------------
# Pairwise head — sees both images at once, emits the BT logit directly.
# ---------------------------------------------------------------------------
def fuse_pairwise(
    text_emb: torch.Tensor,
    img_a: torch.Tensor,
    img_b: torch.Tensor,
) -> torch.Tensor:
    """Pairwise feature fusion: 6 chunks of size ``emb_dim`` each.

    The independent-scoring path has to discover that ``image_a`` and
    ``image_b`` should be compared along directions that align with the
    *prompt* — but it sees only one image at a time, so the comparison
    happens implicitly via two scalar outputs.  Here we put the
    difference into the head's input directly so the head can learn
    prompt-conditioned discriminative features without first projecting
    each image to a scalar.

    Components: ``[text, img_a, img_b, img_a − img_b, |img_a − img_b|,
    text * (img_a − img_b)]`` — the last term cross-couples the prompt
    with the per-image delta, which is the geometric quantity that
    actually decides the BT outcome.
    """
    diff = img_a - img_b
    return torch.cat(
        [text_emb, img_a, img_b, diff, diff.abs(), text_emb * diff],
        dim=-1,
    )


class PairwiseScoringMLP(nn.Module):
    """A 2-hidden-layer MLP that maps ``(text, img_a, img_b) → BT logit``.

    Counterpart to :class:`ScoringMLP` for the pairwise-fusion experiment.
    Output is a scalar **BT logit** (already representing
    ``score_a − score_b``), so the training loop applies ``logit_scale``
    directly to it without subtracting two scalars.

    To enforce antisymmetry under (img_a ↔ img_b), train with random
    A/B order in the data (which preprocess_data.py already does); the
    head learns approximate symmetry from this.  A stricter alternative
    is the symmetrized form ``½ · (f(t, a, b) − f(t, b, a))`` — see
    :meth:`forward_symmetric` below.
    """

    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        use_input_layernorm: bool = True,
    ):
        super().__init__()
        in_dim = 6 * emb_dim
        self.use_input_layernorm = use_input_layernorm
        self.input_norm = nn.LayerNorm(in_dim) if use_input_layernorm else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        text_emb: torch.Tensor,
        img_a: torch.Tensor,
        img_b: torch.Tensor,
    ) -> torch.Tensor:
        x = fuse_pairwise(text_emb, img_a, img_b)
        x = self.input_norm(x)
        return self.net(x).squeeze(-1)

    def forward_symmetric(
        self,
        text_emb: torch.Tensor,
        img_a: torch.Tensor,
        img_b: torch.Tensor,
    ) -> torch.Tensor:
        """Antisymmetric variant: ``½ · (f(t, a, b) − f(t, b, a))``.

        Doubles forward compute but guarantees ``score(a, b) = −score(b, a)``.
        Useful as a diagnostic / robustness check.
        """
        return 0.5 * (self(text_emb, img_a, img_b) - self(text_emb, img_b, img_a))


class PairwiseMultiHeadScorer(nn.Module):
    """One :class:`PairwiseScoringMLP` per dimension.

    API mirror of :class:`MultiHeadScorer` but the per-call signature is
    ``score(text, img_a, img_b, dim)`` and the output is the **BT logit
    directly** (not a scalar score to be subtracted later).
    """

    def __init__(
        self,
        dimensions: list[str],
        emb_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        use_input_layernorm: bool = True,
    ):
        super().__init__()
        self.dimensions = sorted(dimensions)
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.use_input_layernorm = use_input_layernorm
        self.heads = nn.ModuleDict({
            dim: PairwiseScoringMLP(
                emb_dim, hidden_dim, dropout,
                use_input_layernorm=use_input_layernorm,
            )
            for dim in self.dimensions
        })

    def pair_logit(
        self,
        text_emb: torch.Tensor,
        img_a: torch.Tensor,
        img_b: torch.Tensor,
        dimension: str,
    ) -> torch.Tensor:
        if dimension not in self.heads:
            raise KeyError(
                f"Unknown dimension '{dimension}'. Known: {list(self.heads.keys())}"
            )
        return self.heads[dimension](text_emb, img_a, img_b)

    def pair_logit_grouped(
        self,
        text_emb: torch.Tensor,
        img_a: torch.Tensor,
        img_b: torch.Tensor,
        dimensions: list[str],
    ) -> torch.Tensor:
        """Grouped variant for a heterogeneous batch (mirror of
        :meth:`MultiHeadScorer.score_grouped`)."""
        if len(dimensions) != text_emb.size(0):
            raise ValueError("len(dimensions) must equal batch size")

        out = torch.empty(text_emb.size(0), device=text_emb.device, dtype=text_emb.dtype)
        unique = sorted(set(dimensions))
        for dim in unique:
            idx = [i for i, d in enumerate(dimensions) if d == dim]
            idx_t = torch.tensor(idx, device=text_emb.device, dtype=torch.long)
            sub_t = text_emb.index_select(0, idx_t)
            sub_a = img_a.index_select(0, idx_t)
            sub_b = img_b.index_select(0, idx_t)
            logits = self.heads[dim](sub_t, sub_a, sub_b)
            out.index_copy_(0, idx_t, logits.to(out.dtype))
        return out

    def state_dict_for_checkpoint(self) -> dict:
        return {
            "state_dict": self.state_dict(),
            "config": {
                "kind": "pairwise",
                "dimensions": self.dimensions,
                "emb_dim": self.emb_dim,
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout,
                "use_input_layernorm": self.use_input_layernorm,
            },
        }

    @classmethod
    def from_checkpoint(cls, payload: dict) -> "PairwiseMultiHeadScorer":
        cfg = payload["config"]
        m = cls(
            dimensions=cfg["dimensions"],
            emb_dim=cfg["emb_dim"],
            hidden_dim=cfg.get("hidden_dim", 512),
            dropout=cfg.get("dropout", 0.1),
            use_input_layernorm=cfg.get("use_input_layernorm", True),
        )
        m.load_state_dict(payload["state_dict"])
        return m
