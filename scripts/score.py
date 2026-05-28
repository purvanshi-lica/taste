#!/usr/bin/env python
"""Score a CSV of image pairs with a trained TASTE checkpoint.

Thin wrapper around ``taste_scorer.cli.main`` so the package can be run
without installing it (``python scripts/score.py score ...``).  Once the
package is installed with ``pip install -e .``, the ``taste-score``
console script does the same thing.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make ``src/taste_scorer`` importable when running from a clone without
# ``pip install``.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from taste_scorer.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
