"""InstantHMR inference only, vendored for RGB MoCap's live body estimator.

Upstream: https://github.com/Naturalpad-studio/InstantHMR (Apache-2.0), commit
5437400d4b25c1972821e51916fac34bd96acce6. `inference.py` and `models.py` are
unmodified. This `__init__.py` replaces upstream's, which also imports the
RF-DETR person detector, the Rerun visualiser and the demo pipeline; RGB MoCap
uses its own detector and viewer, so those modules are not included.
"""
from .inference import HMRPrediction, InstantHMR

__all__ = ["HMRPrediction", "InstantHMR"]
