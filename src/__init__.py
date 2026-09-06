"""
Talaix Platform - Core Source Package.

An integrated 3-layer system combining DeepTech, AI/satellite data, and
environmental protection to prevent wildfires via subsurface hydration barriers.

Modules:
    - prediction:      Fire risk & spread prediction (ML + physics-based models,
                       Canadian FWI fire-danger system).
    - gis_mapping:     Earth Observation ingestion (real Sentinel-2 via STAC,
                       ESA WorldCover land cover), GIS processing, and mapping.
    - hydration_control: Protection optimisation and adaptive water intervention planning.
    - dashboard:       Interactive dashboard, REST API, real-data analysis engine,
                       caching, and alert watches.
    - standard_formats: Standard data format support (GeoJSON, GML, CSV).
"""

__version__ = "1.0.0"
__all__ = [
    "prediction",
    "gis_mapping",
    "hydration_control",
    "dashboard",
    "security",
]