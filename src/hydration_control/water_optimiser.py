"""
Water resource optimisation for protection interventions.

Implements the Water-Use Efficiency Ratio (WUER) and water-scarce resource
allocation logic.

Key equation:
    WUER = (Risk_baseline - Risk_Talaix) / Volume of water applied
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class WaterUseEfficiency:
    """Container for a WUER calculation result."""

    risk_baseline: float
    risk_hydrashield: float
    water_volume_m3: float
    wuer: float

    def to_dict(self) -> Dict[str, float]:
        """Return WUER result as a dictionary."""
        return {
            "risk_baseline": self.risk_baseline,
            "risk_hydrashield": self.risk_hydrashield,
            "water_volume_m3": self.water_volume_m3,
            "wuer": self.wuer,
        }


@dataclass
class WaterOptimiser:
    """
    Optimise water allocation across protection zones.

    Parameters
    ----------
    water_available_m3 : float
        Total available water volume (m^3) in water-scarce mode.
    """

    water_available_m3: float = 1000.0

    def compute_wuer(
        self,
        risk_baseline: float,
        risk_hydrashield: float,
        water_volume_m3: float,
    ) -> WaterUseEfficiency:
        """
        Compute the Water-Use Efficiency Ratio.

            WUER = (Risk_baseline - Risk_Talaix) / Volume of water applied

        Parameters
        ----------
        risk_baseline : float
            Baseline risk (e.g., expected burned area or probability).
        risk_hydrashield : float
            Risk after Talaix intervention.
        water_volume_m3 : float
            Volume of water applied (m^3).

        Returns
        -------
        WaterUseEfficiency
            WUER result.
        """
        risk_reduction = risk_baseline - risk_hydrashield
        if water_volume_m3 <= 0:
            wuer = 0.0
        else:
            wuer = risk_reduction / water_volume_m3
        return WaterUseEfficiency(
            risk_baseline=float(risk_baseline),
            risk_hydrashield=float(risk_hydrashield),
            water_volume_m3=float(water_volume_m3),
            wuer=float(wuer),
        )

    def allocate_water(
        self,
        zone_priorities: List[float],
        zone_areas_m2: List[float],
        application_rate_m3_per_m2: float = 0.01,
    ) -> List[float]:
        """
        Allocate available water across zones by priority (water-scarce mode).

        Higher-priority zones (e.g., hospitals, schools, single-access
        evacuation routes) receive water first until the budget is exhausted.

        Parameters
        ----------
        zone_priorities : List[float]
            Priority scores for each zone (higher = more critical).
        zone_areas_m2 : List[float]
            Area of each zone (m^2).
        application_rate_m3_per_m2 : float
            Water application rate (m^3 per m^2).

        Returns
        -------
        List[float]
            Water volume allocated to each zone (m^3).
        """
        n = len(zone_priorities)
        if n == 0:
            return []

        # Sort zones by priority (descending)
        order = np.argsort(-np.asarray(zone_priorities, dtype=float))
        allocations = np.zeros(n, dtype=float)
        remaining = self.water_available_m3

        for idx in order:
            required = zone_areas_m2[idx] * application_rate_m3_per_m2
            if remaining <= 0:
                break
            allocated = min(required, remaining)
            allocations[idx] = allocated
            remaining -= allocated

        return allocations.tolist()

    def water_savings(
        self,
        conventional_volume_m3: float,
        hydrashield_volume_m3: float,
    ) -> float:
        """
        Compute water savings percentage vs. conventional approach.

        Parameters
        ----------
        conventional_volume_m3 : float
            Water volume used by conventional aerial firefighting (m^3).
        hydrashield_volume_m3 : float
            Water volume used by Talaix subsurface hydration (m^3).

        Returns
        -------
        float
            Percentage water savings (0-100).
        """
        if conventional_volume_m3 <= 0:
            return 0.0
        savings = (conventional_volume_m3 - hydrashield_volume_m3) / conventional_volume_m3
        return float(np.clip(savings, 0.0, 1.0) * 100.0)
