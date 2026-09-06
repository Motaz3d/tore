"""
Talaix GIS Mapping Module.

Earth Observation ingestion, GIS processing, and mapping.

Key components:
    - indices:       Vegetation/water indices (NDVI, NDMI, NDWI) computation.
    - data_fusion:   Cloud cover mitigation via Sentinel-1 SAR + ERA5-Land fusion.
    - mapping:       Raster/vector processing and protection zone mapping.
"""

from .indices import compute_ndvi, compute_ndmi, compute_ndwi
from .data_fusion import DataFusionPipeline
from .mapping import ProtectionZoneMapper, ProtectionZone

__all__ = [
    "compute_ndvi",
    "compute_ndmi",
    "compute_ndwi",
    "DataFusionPipeline",
    "ProtectionZoneMapper",
    "ProtectionZone",
]
