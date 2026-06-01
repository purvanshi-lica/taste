"""taste-scorer — inference for the TASTE preference model.

Public API::

    from taste_scorer import PreferenceScorer, compute_leaderboard

    scorer = PreferenceScorer.from_checkpoint("path/to/best_pairwise/best")
    df_out = scorer.score_pairs(df_in, image_dir="path/to/images/")

    # If df_in had 'model_a' / 'model_b' columns, df_out will too —
    # build a per-model × per-dim leaderboard from the scored frame:
    leaderboard = compute_leaderboard(df_out)
"""

from taste_scorer.scorer import (
    PreferenceScorer,
    compute_leaderboard,
    format_leaderboard,
)

__all__ = ["PreferenceScorer", "compute_leaderboard", "format_leaderboard"]
__version__ = "0.2.0"
