"""
Talaix Hydration Control Module.

Protection optimisation and adaptive water intervention planning.

Key components:
    - water_optimiser: Water-Use Efficiency Ratio (WUER) and resource allocation.
    - intervention:    Adaptive water intervention planning (where/when/how much).
    - verification:    Hindcasting validation and feedback loop.
"""

from .water_optimiser import WaterOptimiser, WaterUseEfficiency
from .intervention import InterventionPlanner, InterventionPlan
from .verification import HindcastValidator, HindcastResult, ContinuousVerificationSystem

__all__ = [
    "WaterOptimiser",
    "WaterUseEfficiency",
    "InterventionPlanner",
    "InterventionPlan",
    "HindcastValidator",
    "HindcastResult",
    "ContinuousVerificationSystem",
]