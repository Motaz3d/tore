"""
Adapter over ``src.gis_mapping`` — Earth Observation ingestion and GIS
indices (NDVI/NDMI/NDWI, land cover). Reserved for the spatial TX analysis
level (TX-3); imported lazily so ``tx_core`` stays import-light.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def vegetation_index(band_nir: float, band_red: float) -> Optional[float]:
    """NDVI from reflectance bands via the platform indices module."""
    try:
        from src.gis_mapping.indices import ndvi

        return ndvi(band_nir, band_red)
    except Exception:
        return None


def moisture_index(band_nir: float, band_swir: float) -> Optional[float]:
    """NDMI from reflectance bands via the platform indices module."""
    try:
        from src.gis_mapping.indices import ndmi

        return ndmi(band_nir, band_swir)
    except Exception:
        return None


def landcover_classes() -> List[str]:
    """ESA WorldCover class names available in the platform mapping module."""
    try:
        from src.gis_mapping.landcover import WORLDCOVER_CLASSES

        return list(WORLDCOVER_CLASSES)
    except Exception:
        return []
