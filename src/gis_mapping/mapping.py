"""
Raster/vector processing and protection zone mapping.

Computes the Critical Protection Zone (CPZ) around vulnerable assets and
evacuation corridors, based on fire arrival time and risk thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ProtectionZone:
    """A computed protection zone around an asset."""

    asset_id: str
    asset_type: str
    centroid: Tuple[float, float]
    radius_m: float
    area_m2: float
    risk_level: str  # 'green', 'yellow', 'red'
    fire_arrival_time_min: float

    def to_dict(self) -> Dict[str, object]:
        """Return zone as a dictionary."""
        return {
            "asset_id": self.asset_id,
            "asset_type": self.asset_type,
            "centroid": self.centroid,
            "radius_m": self.radius_m,
            "area_m2": self.area_m2,
            "risk_level": self.risk_level,
            "fire_arrival_time_min": self.fire_arrival_time_min,
        }


@dataclass
class ProtectionZoneMapper:
    """
    Compute Critical Protection Zones around vulnerable assets.

    Parameters
    ----------
    risk_threshold : float
        Probability-of-spread threshold theta for zone inclusion.
    min_lead_time_min : float
        Minimum acceptable fire arrival lead time (minutes).
    """

    risk_threshold: float = 0.5
    min_lead_time_min: float = 60.0

    def compute_zone_radius(
        self,
        fire_arrival_time_min: float,
        ros_m_per_min: float,
    ) -> float:
        """
        Compute the protection zone radius (metres) for a given lead time.

        radius = ROS * lead_time

        Parameters
        ----------
        fire_arrival_time_min : float
            Desired fire arrival lead time (minutes).
        ros_m_per_min : float
            Rate of spread (m/min).

        Returns
        -------
        float
            Protection zone radius in metres.
        """
        return float(max(fire_arrival_time_min, 0.0) * max(ros_m_per_min, 0.0))

    def classify_risk(
        self,
        probability_of_spread: float,
        fire_arrival_time_min: float,
    ) -> str:
        """
        Classify a zone into a traffic-light risk level.

        - 'green': low risk, sufficient lead time.
        - 'yellow': moderate risk, review required.
        - 'red': high risk, hold / manual reassessment.

        Parameters
        ----------
        probability_of_spread : float
            Probability of spread in [0, 1].
        fire_arrival_time_min : float
            Fire arrival lead time (minutes).

        Returns
        -------
        str
            Risk level: 'green', 'yellow', or 'red'.
        """
        if probability_of_spread < self.risk_threshold and fire_arrival_time_min >= self.min_lead_time_min:
            return "green"
        if probability_of_spread < 0.8 and fire_arrival_time_min >= self.min_lead_time_min * 0.5:
            return "yellow"
        return "red"

    def build_protection_zones(
        self,
        assets: List[Dict[str, object]],
        ros_m_per_min: float,
        probability_of_spread: float,
        lead_time_min: float,
    ) -> List[ProtectionZone]:
        """
        Build protection zones for a list of assets.

        Parameters
        ----------
        assets : List[Dict[str, object]]
            Each asset dict must contain 'id', 'type', and 'centroid'
            (a (lon, lat) tuple).
        ros_m_per_min : float
            Rate of spread (m/min).
        probability_of_spread : float
            Probability of spread in [0, 1].
        lead_time_min : float
            Fire arrival lead time (minutes).

        Returns
        -------
        List[ProtectionZone]
            Computed protection zones.
        """
        zones: List[ProtectionZone] = []
        radius = self.compute_zone_radius(lead_time_min, ros_m_per_min)
        risk = self.classify_risk(probability_of_spread, lead_time_min)

        for asset in assets:
            asset_id = str(asset.get("id", "unknown"))
            asset_type = str(asset.get("type", "asset"))
            centroid = tuple(asset.get("centroid", (0.0, 0.0)))

            zones.append(
                ProtectionZone(
                    asset_id=asset_id,
                    asset_type=asset_type,
                    centroid=centroid,
                    radius_m=radius,
                    area_m2=np.pi * radius * radius,
                    risk_level=risk,
                    fire_arrival_time_min=lead_time_min,
                )
            )
        return zones

    def compute_critical_area(
        self,
        zones: List[ProtectionZone],
        max_area_m2: Optional[float] = None,
    ) -> float:
        """
        Compute the total critical protection area.

        Optionally constrained by a maximum available area (water-scarce mode).

        Parameters
        ----------
        zones : List[ProtectionZone]
            Protection zones.
        max_area_m2 : Optional[float]
            Maximum allowable area (m^2).

        Returns
        -------
        float
            Total critical area (m^2).
        """
        total = sum(z.area_m2 for z in zones)
        if max_area_m2 is not None:
            total = min(total, max_area_m2)
        return float(total)

    def compute_3d_protection_zones(
        self,
        assets: List[Dict[str, object]],
        ros_m_per_min: float,
        probability_of_spread: float,
        lead_time_min: float,
        elevation_data: Optional[np.ndarray] = None,
        slope_data: Optional[np.ndarray] = None,
    ) -> List[ProtectionZone]:
        """
        Compute protection zones with 3D terrain considerations.

        Parameters
        ----------
        assets : List[Dict[str, object]]
            Each asset dict must contain 'id', 'type', and 'centroid'.
        ros_m_per_min : float
            Rate of spread (m/min).
        probability_of_spread : float
            Probability of spread in [0, 1].
        lead_time_min : float
            Fire arrival lead time (minutes).
        elevation_data : Optional[np.ndarray]
            Elevation data for terrain analysis.
        slope_data : Optional[np.ndarray]
            Slope data for terrain analysis.

        Returns
        -------
        List[ProtectionZone]
            Computed 3D protection zones.
        """
        zones: List[ProtectionZone] = []
        risk = self.classify_risk(probability_of_spread, lead_time_min)

        for asset in assets:
            asset_id = str(asset.get("id", "unknown"))
            asset_type = str(asset.get("type", "asset"))
            centroid = tuple(asset.get("centroid", (0.0, 0.0)))

            radius = self.compute_zone_radius(lead_time_min, ros_m_per_min)

            if elevation_data is not None and slope_data is not None:
                radius = self._adjust_radius_for_terrain(
                    radius, elevation_data, slope_data, centroid
                )

            zones.append(
                ProtectionZone(
                    asset_id=asset_id,
                    asset_type=asset_type,
                    centroid=centroid,
                    radius_m=radius,
                    area_m2=np.pi * radius * radius,
                    risk_level=risk,
                    fire_arrival_time_min=lead_time_min,
                )
            )
        return zones

    def _adjust_radius_for_terrain(
        self,
        base_radius: float,
        elevation_data: np.ndarray,
        slope_data: np.ndarray,
        centroid: Tuple[float, float],
    ) -> float:
        """
        Adjust protection zone radius based on 3D terrain characteristics.

        This is a simplified placeholder: steep slopes increase the radius
        (fire spreads faster uphill), and elevation affects the radius.

        Parameters
        ----------
        base_radius : float
            Base protection zone radius.
        elevation_data : np.ndarray
            Elevation data for terrain analysis.
        slope_data : np.ndarray
            Slope data for terrain analysis.
        centroid : Tuple[float, float]
            Asset centroid (lon, lat).

        Returns
        -------
        float
            Adjusted protection zone radius based on terrain.
        """
        slope_factor = 1.0
        elevation_factor = 1.0

        # Use simple statistics as a placeholder for georeferenced lookup.
        if slope_data.size > 0:
            mean_slope = float(np.mean(slope_data))
            slope_factor = 1.0 + 0.01 * max(mean_slope, 0.0)

        if elevation_data.size > 0:
            mean_elevation = float(np.mean(elevation_data))
            elevation_factor = 1.0 + 0.0001 * max(mean_elevation, 0.0)

        return base_radius * slope_factor * elevation_factor

    def compute_evacuation_routes(
        self,
        assets: List[Dict[str, object]],
        start_points: List[Tuple[float, float]],
        end_points: List[Tuple[float, float]],
        terrain_data: Optional[np.ndarray] = None,
        road_network: Optional[object] = None,
    ) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Compute optimal evacuation routes using network analysis.

        This simplified implementation returns straight-line routes. A full
        implementation would use routing algorithms (e.g., Dijkstra or A*)
        over a road network.

        Parameters
        ----------
        assets : List[Dict[str, object]]
            Assets that need evacuation routes.
        start_points : List[Tuple[float, float]]
            Starting points for evacuation (near assets).
        end_points : List[Tuple[float, float]]
            Safe destination points for evacuation.
        terrain_data : Optional[np.ndarray]
            Terrain data affecting evacuation routes.
        road_network : Optional[object]
            Existing road network to consider (e.g., a GeoDataFrame).

        Returns
        -------
        List[Tuple[Tuple[float, float], Tuple[float, float]]]
            Optimal evacuation routes as (start, end) pairs.
        """
        routes = []
        for start, end in zip(start_points, end_points):
            routes.append((start, end))
        return routes

    def analyze_wind_direction_impact(
        self,
        base_zones: List[ProtectionZone],
        wind_direction_deg: float,
        wind_speed_kmh: float,
        terrain_orientation: Optional[np.ndarray] = None,
    ) -> List[Dict[str, object]]:
        """
        Analyze the impact of wind direction on protection zone effectiveness.

        Fire spreads faster downwind, so protection zones may need to be
        extended in the downwind direction and compressed crosswind.

        Parameters
        ----------
        base_zones : List[ProtectionZone]
            Base protection zones.
        wind_direction_deg : float
            Wind direction in degrees (0=N, 90=E, etc.).
        wind_speed_kmh : float
            Wind speed in km/h.
        terrain_orientation : Optional[np.ndarray]
            Aspect data for terrain orientation analysis.

        Returns
        -------
        List[Dict[str, object]]
            Wind-adjusted zone descriptors including major/minor axes.
        """
        wind_speed_kmh = max(wind_speed_kmh, 0.0)

        adjusted = []
        for zone in base_zones:
            major_axis = zone.radius_m * (1 + 0.3 * wind_speed_kmh / 100.0)
            minor_axis = zone.radius_m * (1 - 0.1 * wind_speed_kmh / 100.0)
            minor_axis = max(minor_axis, zone.radius_m * 0.5)

            adjusted.append(
                {
                    "asset_id": zone.asset_id,
                    "centroid": zone.centroid,
                    "radius_m": zone.radius_m,
                    "major_axis_m": float(major_axis),
                    "minor_axis_m": float(minor_axis),
                    "wind_direction_deg": wind_direction_deg,
                    "wind_speed_kmh": wind_speed_kmh,
                    "risk_level": zone.risk_level,
                }
            )
        return adjusted