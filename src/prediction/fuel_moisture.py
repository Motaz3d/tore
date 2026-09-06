"""
Fuel Moisture Content (FMC) estimation and prediction.

Implements the Soil-to-Fuel Moisture Transfer Problem: modelling the capillary
transfer linking subsurface soil moisture to critical surface litter moisture.

The module provides:
    - Estimation of FMC from Earth Observation indices (NDMI, NDWI).
    - A physics-informed capillary transfer model linking soil moisture to
      surface fuel moisture.
    - The Minimum Effective FMC Increase (MEFMI) calculation:
          MEFMI = FMC_target - FMC_baseline
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class FuelMoistureModel:
    """
    Model for estimating and predicting Fuel Moisture Content (FMC).

    Parameters
    ----------
    dead_fuel_time_lag_hours : float
        Time lag (hours) for dead fuel moisture response (1h, 10h, 100h fuels).
    capillary_rise_coefficient : float
        Empirical coefficient linking subsurface soil moisture to surface
        litter moisture (0 < k < 1). Represents the efficiency of capillary
        transfer in the vadose zone.
    """

    dead_fuel_time_lag_hours: float = 10.0
    capillary_rise_coefficient: float = 0.35

    def estimate_fmc_from_ndmi(self, ndmi: np.ndarray) -> np.ndarray:
        """
        Estimate Fuel Moisture Content from the Normalized Difference
        Moisture Index (NDMI).

        A linear calibration is used as a first-order proxy:
            FMC ~= a * NDMI + b

        Parameters
        ----------
        ndmi : np.ndarray
            Normalized Difference Moisture Index values in [-1, 1].

        Returns
        -------
        np.ndarray
            Estimated FMC in percent (0-100).
        """
        ndmi = np.asarray(ndmi, dtype=float)
        # Calibration coefficients (Phase 1 placeholder, to be fitted on data)
        a = 120.0
        b = 30.0
        fmc = a * ndmi + b
        return np.clip(fmc, 0.0, 100.0)

    def estimate_fmc_from_ndwi(self, ndwi: np.ndarray) -> np.ndarray:
        """
        Estimate Fuel Moisture Content from the Normalized Difference
        Water Index (NDWI).

        Parameters
        ----------
        ndwi : np.ndarray
            Normalized Difference Water Index values in [-1, 1].

        Returns
        -------
        np.ndarray
            Estimated FMC in percent (0-100).
        """
        ndwi = np.asarray(ndwi, dtype=float)
        a = 110.0
        b = 35.0
        fmc = a * ndwi + b
        return np.clip(fmc, 0.0, 100.0)

    def capillary_transfer(
        self,
        soil_moisture: np.ndarray,
        soil_moisture_saturation: float = 0.45,
    ) -> np.ndarray:
        """
        Model the capillary transfer of moisture from subsurface soil to
        surface litter fuel.

        The surface fuel moisture responds to subsurface soil moisture with a
        time lag and an efficiency factor (capillary_rise_coefficient).

        Parameters
        ----------
        soil_moisture : np.ndarray
            Subsurface volumetric soil moisture (m^3/m^3).
        soil_moisture_saturation : float
            Saturation soil moisture value used for normalisation.

        Returns
        -------
        np.ndarray
            Resulting surface fuel moisture content (percent).
        """
        soil_moisture = np.asarray(soil_moisture, dtype=float)
        # Normalise soil moisture to a 0-1 relative saturation
        rel_sat = np.clip(soil_moisture / soil_moisture_saturation, 0.0, 1.0)
        # Surface FMC responds proportionally to capillary transfer efficiency
        surface_fmc = 100.0 * self.capillary_rise_coefficient * rel_sat
        return np.clip(surface_fmc, 0.0, 100.0)

    def minimum_effective_fmc_increase(
        self,
        fmc_baseline: float,
        fmc_target: float,
    ) -> float:
        """
        Compute the Minimum Effective FMC Increase (MEFMI).

            MEFMI = FMC_target - FMC_baseline

        Parameters
        ----------
        fmc_baseline : float
            Baseline fuel moisture content (percent).
        fmc_target : float
            Target fuel moisture content required to reduce spread (percent).

        Returns
        -------
        float
            The minimum effective FMC increase (percent).
        """
        return float(fmc_target - fmc_baseline)

    def time_to_reach_target(
        self,
        fmc_baseline: float,
        fmc_target: float,
        soil_moisture: float,
        soil_moisture_saturation: float = 0.45,
    ) -> float:
        """
        Estimate the lead time (hours) required for capillary transfer to raise
        surface FMC from baseline to target.

        Uses a first-order exponential response with the dead fuel time lag.

        Parameters
        ----------
        fmc_baseline : float
            Baseline surface FMC (percent).
        fmc_target : float
            Target surface FMC (percent).
        soil_moisture : float
            Subsurface soil moisture (m^3/m^3).
        soil_moisture_saturation : float
            Saturation soil moisture (m^3/m^3).

        Returns
        -------
        float
            Estimated lead time in hours. Returns inf if the target is not
            achievable given the soil moisture.
        """
        achievable = self.capillary_transfer(
            np.array([soil_moisture]), soil_moisture_saturation
        )[0]
        if achievable <= fmc_baseline:
            return float("inf")

        # First-order approach to equilibrium
        delta = fmc_target - fmc_baseline
        max_delta = achievable - fmc_baseline
        if delta <= 0:
            return 0.0
        if delta >= max_delta:
            return float("inf")

        # t = -tau * ln(1 - delta/max_delta)
        tau = self.dead_fuel_time_lag_hours
        t = -tau * np.log(1.0 - delta / max_delta)
        return float(max(t, 0.0))
