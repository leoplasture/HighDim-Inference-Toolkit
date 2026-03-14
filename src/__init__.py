"""HighDim-Inference-Toolkit.

This package contains:
- Lasso via coordinate descent (`LassoCD`)
- Debiased Lasso for inference (`DebiasedLasso`)
- Trans-Lasso for transfer learning (`TransLasso`)
- Confidence interval helpers (`HighDimCI`)
"""

from .confidence_interval import HighDimCI
from .debiased_lasso import DebiasedLasso
from .lasso import LassoCD
from .trans_lasso import TransLasso

__all__ = [
    "LassoCD",
    "DebiasedLasso",
    "TransLasso",
    "HighDimCI",
]
