"""
Adapter over ``src.prediction`` — fire-weather (FWI), risk model and spread
screening. Reserved for the deterministic/statistical TX analysis levels
(TX-1/TX-2); imported lazily so ``tx_core`` stays import-light.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def fwi_compute(tmax_c: float, rh: float, wind_kmh: float, rain_mm: float,
                ffmc0: Optional[float] = None, **kw: Any) -> Dict[str, Any]:
    """One-day Canadian FWI computation via the platform adapter.

    Returns the FWI-family dict (FFMC/DMC/DC/ISI/BUI/FWI/DSR + danger class)
    or an honest error payload when the module is unavailable.
    """
    try:
        from src.prediction.fwi import FWIAdapter

        return FWIAdapter.compute(
            tmax_c=tmax_c, rh=rh, wind_kmh=wind_kmh, rain_mm=rain_mm,
            ffmc0=ffmc0, **kw,
        )
    except Exception as exc:  # noqa: BLE001 — honest unavailability
        return {"status": "unavailable", "reason": str(exc)}


def risk_model_class() -> Optional[Any]:
    """The composite wildfire risk model class (lazy)."""
    try:
        from src.prediction.risk_model import WildfireRiskModel

        return WildfireRiskModel
    except Exception:
        return None
