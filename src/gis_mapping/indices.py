"""
Computation of vegetation and water indices from multispectral imagery.

Implements standard spectral indices used for fuel moisture and vegetation
health assessment from Copernicus Sentinel-2 data:

    - NDVI : Normalized Difference Vegetation Index
    - NDMI : Normalized Difference Moisture Index
    - NDWI : Normalized Difference Water Index
"""

from __future__ import annotations

from typing import Union

import numpy as np

ArrayLike = Union[np.ndarray, list, tuple]


def _safe_normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute (a - b) / (a + b) with safe handling of zero denominators.

    Parameters
    ----------
    a : np.ndarray
        Numerator band (e.g., NIR).
    b : np.ndarray
        Denominator band (e.g., RED).

    Returns
    -------
    np.ndarray
        Normalized difference index in [-1, 1].
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denom != 0, (a - b) / denom, 0.0)
    return np.clip(result, -1.0, 1.0)


def compute_ndvi(nir: ArrayLike, red: ArrayLike) -> np.ndarray:
    """
    Compute the Normalized Difference Vegetation Index.

        NDVI = (NIR - RED) / (NIR + RED)

    Parameters
    ----------
    nir : ArrayLike
        Near-infrared reflectance band (Sentinel-2 B8).
    red : ArrayLike
        Red reflectance band (Sentinel-2 B4).

    Returns
    -------
    np.ndarray
        NDVI values in [-1, 1].
    """
    return _safe_normalized_difference(np.asarray(nir), np.asarray(red))


def compute_ndmi(nir: ArrayLike, swir: ArrayLike) -> np.ndarray:
    """
    Compute the Normalized Difference Moisture Index.

        NDMI = (NIR - SWIR) / (NIR + SWIR)

    NDMI is sensitive to vegetation water content and is a proven proxy for
    baseline Fuel Moisture Content (FMC).

    Parameters
    ----------
    nir : ArrayLike
        Near-infrared reflectance band (Sentinel-2 B8).
    swir : ArrayLike
        Short-wave infrared band (Sentinel-2 B11).

    Returns
    -------
    np.ndarray
        NDMI values in [-1, 1].
    """
    return _safe_normalized_difference(np.asarray(nir), np.asarray(swir))


def compute_ndwi(green: ArrayLike, nir: ArrayLike) -> np.ndarray:
    """
    Compute the Normalized Difference Water Index.

        NDWI = (GREEN - NIR) / (GREEN + NIR)

    Parameters
    ----------
    green : ArrayLike
        Green reflectance band (Sentinel-2 B3).
    nir : ArrayLike
        Near-infrared reflectance band (Sentinel-2 B8).

    Returns
    -------
    np.ndarray
        NDWI values in [-1, 1].
    """
    return _safe_normalized_difference(np.asarray(green), np.asarray(nir))
