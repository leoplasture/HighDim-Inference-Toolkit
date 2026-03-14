"""HighDim-Inference-Toolkit.

Public API:
- Lasso via coordinate descent: `LassoCD`
- Debiased Lasso inference: `DebiasedLasso`
- Trans-Lasso transfer learning: `TransLasso`
- Confidence interval helpers: `HighDimCI`
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

__version__ = "0.1.0"
