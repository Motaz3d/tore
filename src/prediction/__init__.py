"""
Talaix Prediction Module.

Fire risk & spread prediction combining Machine Learning with established
landscape-scale physics-based wildfire spread models.

Key components:
    - fuel_moisture:  Fuel Moisture Content (FMC) estimation and prediction.
    - fire_spread:    Rate of Spread (ROS) and fire propagation modelling.
    - risk_model:     ML-based wildfire risk assessment.
"""

from .fuel_moisture import FuelMoistureModel
from .fire_spread import FireSpreadModel, RateOfSpread
from .risk_model import WildfireRiskModel, AdvancedWildfireRiskModel

__all__ = [
    "FuelMoistureModel",
    "FireSpreadModel",
    "RateOfSpread",
    "WildfireRiskModel",
]
