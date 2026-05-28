"""Checkpoint-free smoke tests for the taste-scorer package.

These do not require a trained checkpoint, a GPU, or the dataset. Tests that
need torch are skipped automatically if torch is not installed, so the import
surface can still be checked in a minimal environment.

Run with:  pytest -q
"""

from __future__ import annotations

import pytest


def test_package_imports():
    import taste_scorer

    assert hasattr(taste_scorer, "PreferenceScorer")
    assert isinstance(taste_scorer.__version__, str)


def test_public_api_is_minimal():
    import taste_scorer

    # The public surface is intentionally just the scorer.
    assert taste_scorer.__all__ == ["PreferenceScorer"]


def test_cli_entrypoint_importable():
    from taste_scorer import cli

    assert hasattr(cli, "main")


def test_pairwise_head_forward_shape():
    torch = pytest.importorskip("torch")
    from taste_scorer.heads import PairwiseMultiHeadScorer

    emb_dim = 16
    dims = ["preference", "typography"]
    head = PairwiseMultiHeadScorer(dimensions=dims, emb_dim=emb_dim, hidden_dim=8)
    head.eval()

    batch = 4
    t = torch.randn(batch, emb_dim)
    i_a = torch.randn(batch, emb_dim)
    i_b = torch.randn(batch, emb_dim)

    with torch.no_grad():
        logit = head.pair_logit(t, i_a, i_b, "preference")

    assert logit.shape[0] == batch
