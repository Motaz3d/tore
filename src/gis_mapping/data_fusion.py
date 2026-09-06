"""
Data fusion pipeline for cloud cover mitigation.

To address Sentinel-2 cloud cover limitations, this pipeline fuses:
    - Sentinel-1 SAR (all-weather, day/night) for soil moisture estimation.
    - ECMWF soil moisture reanalysis (ERA5-Land) for temporal interpolation.
    - Additional data sources as available (MODIS fire detection, ESA wind data).

The result is a gap-filled, cloud-free soil moisture / FMC product suitable
for downstream risk modelling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class DataFusionPipeline:
    """
    Fuse multiple EO data sources to produce a cloud-free moisture product.

    Parameters
    ----------
    sar_weight : float
        Weight given to Sentinel-1 SAR-derived soil moisture (0-1).
    reanalysis_weight : float
        Weight given to ERA5-Land reanalysis soil moisture (0-1).
    """

    sar_weight: float = 0.5
    reanalysis_weight: float = 0.5

    def __post_init__(self) -> None:
        total = self.sar_weight + self.reanalysis_weight
        if total <= 0:
            raise ValueError("Weights must sum to a positive value.")
        # Normalise weights
        self.sar_weight = self.sar_weight / total
        self.reanalysis_weight = self.reanalysis_weight / total

    def weather_impact_factor(self, weather_conditions: Dict[str, float]) -> float:
        """
        Calculate a factor that adjusts weights based on weather conditions.
        
        Parameters
        ----------
        weather_conditions : Dict[str, float]
            Dictionary containing weather parameters like precipitation, humidity, etc.
        
        Returns
        -------
        float
            Weather impact factor in [0, 2] range (values > 1 increase weight).
        """
        # Default to neutral factor
        factor = 1.0
        
        # Precipitation typically affects SAR less than reanalysis
        precipitation = weather_conditions.get('precipitation', 0.0)
        if precipitation > 5.0:  # Heavy rain threshold
            # Increase SAR weight since it's more reliable during heavy precipitation
            factor *= 1.2
        elif precipitation < 1.0:  # Dry conditions
            # Reanalysis might be more reliable in dry conditions
            factor *= 0.9
            
        # Humidity affects soil moisture retrieval differently
        humidity = weather_conditions.get('humidity', 0.5)
        if humidity > 0.8:  # High humidity
            factor *= 0.95  # Both methods might be affected differently
            
        return min(max(factor, 0.1), 2.0)  # Clamp between 0.1 and 2.0

    def terrain_visibility_factor(self, terrain_type: str) -> float:
        """
        Calculate a factor that adjusts weights based on terrain characteristics.
        
        Parameters
        ----------
        terrain_type : str
            Type of terrain (e.g., 'forest', 'urban', 'agricultural', 'mountain').
        
        Returns
        -------
        float
            Terrain visibility factor in [0, 2] range.
        """
        # Different terrains affect SAR and reanalysis differently
        terrain_factors = {
            'forest': 1.1,      # SAR performs well in forests
            'urban': 1.0,       # Balanced performance
            'agricultural': 0.9, # Slightly favor reanalysis
            'mountain': 1.2,    # SAR handles terrain variations well
            'desert': 0.8,      # SAR may be less reliable in arid regions
            'water': 1.0,       # Mixed performance
            'mixed': 1.0        # Default mixed terrain
        }
        
        return terrain_factors.get(terrain_type.lower(), 1.0)

    def adaptive_fusion_weights(self, 
                               sar_data_quality: float, 
                               reanalysis_data_quality: float, 
                               weather_conditions: Dict[str, float],
                               terrain_type: str) -> Tuple[float, float]:
        """
        Calculate weights dynamically based on data quality, weather, and terrain.
        
        Parameters
        ----------
        sar_data_quality : float
            Quality score for SAR data (0-1).
        reanalysis_data_quality : float
            Quality score for reanalysis data (0-1).
        weather_conditions : Dict[str, float]
            Current weather conditions.
        terrain_type : str
            Type of terrain.
        
        Returns
        -------
        Tuple[float, float]
            Adjusted SAR and reanalysis weights that sum to 1.0.
        """
        # Base quality ratio
        total_quality = sar_data_quality + reanalysis_data_quality
        if total_quality <= 0:
            # Fallback to original weights if quality data unavailable
            return self.sar_weight, self.reanalysis_weight
            
        base_sar_weight = sar_data_quality / total_quality
        base_reanalysis_weight = reanalysis_data_quality / total_quality
        
        # Apply weather and terrain factors
        weather_factor = self.weather_impact_factor(weather_conditions)
        terrain_factor = self.terrain_visibility_factor(terrain_type)
        
        # Adjust weights based on factors
        adjusted_sar_weight = base_sar_weight * weather_factor * terrain_factor
        adjusted_reanalysis_weight = base_reanalysis_weight * weather_factor  # Terrain factor mainly affects SAR
        
        # Normalize to sum to 1.0
        total = adjusted_sar_weight + adjusted_reanalysis_weight
        if total <= 0:
            return self.sar_weight, self.reanalysis_weight  # Fallback
            
        return adjusted_sar_weight / total, adjusted_reanalysis_weight / total

    def fuse_soil_moisture(
        self,
        sar_soil_moisture: np.ndarray,
        reanalysis_soil_moisture: np.ndarray,
        cloud_mask: Optional[np.ndarray] = None,
        sar_data_quality: Optional[float] = 0.8,
        reanalysis_data_quality: Optional[float] = 0.8,
        weather_conditions: Optional[Dict[str, float]] = None,
        terrain_type: str = 'mixed',
    ) -> np.ndarray:
        """
        Fuse SAR and reanalysis soil moisture into a single product.

        Where cloud cover is present (cloud_mask == True), the SAR/reanalysis
        fusion fills the gap. Where no cloud is present, the optical-derived
        value is preferred.

        Parameters
        ----------
        sar_soil_moisture : np.ndarray
            Soil moisture from Sentinel-1 SAR (m^3/m^3).
        reanalysis_soil_moisture : np.ndarray
            Soil moisture from ERA5-Land reanalysis (m^3/m^3).
        cloud_mask : Optional[np.ndarray]
            Boolean mask where True indicates cloud-covered pixels.
        sar_data_quality : Optional[float]
            Quality score for SAR data (0-1).
        reanalysis_data_quality : Optional[float]
            Quality score for reanalysis data (0-1).
        weather_conditions : Optional[Dict[str, float]]
            Current weather conditions.
        terrain_type : str
            Type of terrain.

        Returns
        -------
        np.ndarray
            Fused soil moisture product (m^3/m^3).
        """
        sar = np.asarray(sar_soil_moisture, dtype=float)
        rean = np.asarray(reanalysis_soil_moisture, dtype=float)

        if weather_conditions is None:
            weather_conditions = {}

        # Calculate adaptive weights based on data quality, weather, and terrain
        sar_weight, reanalysis_weight = self.adaptive_fusion_weights(
            sar_data_quality or 0.8,
            reanalysis_data_quality or 0.8,
            weather_conditions,
            terrain_type
        )

        # Weighted fusion of SAR and reanalysis
        fused = sar_weight * sar + reanalysis_weight * rean

        if cloud_mask is not None:
            cloud_mask = np.asarray(cloud_mask, dtype=bool)
            # Where no cloud, prefer the (higher-confidence) optical estimate.
            # Here we use the reanalysis as the "clear" reference.
            fused = np.where(cloud_mask, fused, rean)

        return np.clip(fused, 0.0, 1.0)

    def temporal_interpolate(
        self,
        series: np.ndarray,
        timestamps: np.ndarray,
        target_timestamps: np.ndarray,
    ) -> np.ndarray:
        """
        Linearly interpolate a time series to target timestamps.

        Parameters
        ----------
        series : np.ndarray
            Observed values at `timestamps`.
        timestamps : np.ndarray
            Observation times (numeric, e.g., hours since epoch).
        target_timestamps : np.ndarray
            Times at which to interpolate.

        Returns
        -------
        np.ndarray
            Interpolated values at target timestamps.
        """
        series = np.asarray(series, dtype=float)
        timestamps = np.asarray(timestamps, dtype=float)
        target_timestamps = np.asarray(target_timestamps, dtype=float)
        return np.interp(target_timestamps, timestamps, series)

    def estimate_fmc_from_fused_moisture(
        self,
        fused_soil_moisture: np.ndarray,
        soil_moisture_saturation: float = 0.45,
        capillary_coefficient: float = 0.35,
    ) -> np.ndarray:
        """
        Convert fused soil moisture to a surface fuel moisture estimate.

        Parameters
        ----------
        fused_soil_moisture : np.ndarray
            Fused soil moisture (m^3/m^3).
        soil_moisture_saturation : float
            Saturation soil moisture (m^3/m^3).
        capillary_coefficient : float
            Capillary transfer efficiency (0-1).

        Returns
        -------
        np.ndarray
            Estimated surface FMC (percent).
        """
        fused = np.asarray(fused_soil_moisture, dtype=float)
        rel_sat = np.clip(fused / soil_moisture_saturation, 0.0, 1.0)
        fmc = 100.0 * capillary_coefficient * rel_sat
        return np.clip(fmc, 0.0, 100.0)
