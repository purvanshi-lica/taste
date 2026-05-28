"""taste-scorer — inference for the TASTE preference model.

Public API::

    from taste_scorer import PreferenceScorer

    scorer = PreferenceScorer.from_checkpoint("path/to/best_pairwise/best")
    df_out = scorer.score_pairs(df_in, image_dir="path/to/images/")
"""

from taste_scorer.scorer import PreferenceScorer

__all__ = ["PreferenceScorer"]
__version__ = "0.1.0"
