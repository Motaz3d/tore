"""
Canadian Forest Fire Weather Index (FWI) System.

Implements the standard Van Wagner (1987) equations used worldwide — and by
EFFIS / the Copernicus Emergency Management Service — for fire danger rating:

    FFMC  Fine Fuel Moisture Code       (surface litter, fast response)
    DMC   Duff Moisture Code            (medium-depth organic layer)
    DC    Drought Code                  (deep organic layer, slow response)
    ISI   Initial Spread Index          (FFMC x wind)
    BUI   Buildup Index                 (DMC x DC)
    FWI   Fire Weather Index            (ISI x BUI)
    DSR   Daily Severity Rating

Inputs are daily values representative of local noon:
    - temperature (degC)
    - relative humidity (%)
    - 10 m wind speed (km/h)
    - 24 h precipitation (mm)

When only daily aggregates are available (e.g. Open-Meteo daily series), the
standard screening approximation is T_max / RH_min / mean wind / rain sum.
That approximation is declared in the provenance metadata of the analysis.

Day-length tables are the northern-hemisphere values (valid for Europe).
Danger classes follow the EFFIS thresholds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

# Day-length factors (northern hemisphere), Van Wagner (1987).
_DMC_DAY_LENGTH = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]
_DC_DAY_LENGTH = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]

# Conventional startup values for the codes.
DEFAULT_SEED = {"ffmc": 85.0, "dmc": 6.0, "dc": 15.0}

# EFFIS fire danger classes (FWI thresholds).
EFFIS_CLASSES = [
    (5.2, "Very low"),
    (11.2, "Low"),
    (21.3, "Moderate"),
    (38.0, "High"),
    (50.0, "Very high"),
    (math.inf, "Extreme"),
]

# Simplified 4-class scale used for user-facing output.
SIMPLE_CLASSES = [
    (11.2, "Low"),
    (21.3, "Moderate"),
    (38.0, "High"),
    (math.inf, "Extreme"),
]


@dataclass
class FWIDay:
    """One day of FWI System output."""

    date: str
    ffmc: float
    dmc: float
    dc: float
    isi: float
    bui: float
    fwi: float
    dsr: float
    danger_class: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "date": self.date,
            "ffmc": round(self.ffmc, 1),
            "dmc": round(self.dmc, 1),
            "dc": round(self.dc, 1),
            "isi": round(self.isi, 1),
            "bui": round(self.bui, 1),
            "fwi": round(self.fwi, 1),
            "dsr": round(self.dsr, 2),
            "danger_class": self.danger_class,
        }


def _ffmc(temp: float, rh: float, wind: float, rain: float, ffmc_prev: float) -> float:
    """Fine Fuel Moisture Code for one day."""
    mo = 147.2 * (101.0 - ffmc_prev) / (59.5 + ffmc_prev)

    if rain > 0.5:
        rf = rain - 0.5
        if mo > 150.0:
            mo = (
                mo
                + 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1.0 - math.exp(-6.93 / rf))
                + 0.0015 * (mo - 150.0) ** 2 * math.sqrt(rf)
            )
        else:
            mo = mo + 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1.0 - math.exp(-6.93 / rf))
        mo = min(mo, 250.0)

    rh_c = min(max(rh, 0.0), 100.0)
    ed = (
        0.942 * rh_c ** 0.679
        + 11.0 * math.exp((rh_c - 100.0) / 10.0)
        + 0.18 * (21.1 - temp) * (1.0 - math.exp(-0.115 * rh_c))
    )
    ew = (
        0.618 * rh_c ** 0.753
        + 10.0 * math.exp((rh_c - 100.0) / 10.0)
        + 0.18 * (21.1 - temp) * (1.0 - math.exp(-0.115 * rh_c))
    )

    if mo > ed:
        ko = 0.424 * (1.0 - (rh_c / 100.0) ** 1.7) + 0.0694 * math.sqrt(max(wind, 0.0)) * (
            1.0 - (rh_c / 100.0) ** 8
        )
        kd = ko * 0.581 * math.exp(0.0365 * temp)
        m = ed + (mo - ed) * 10.0 ** (-kd)
    elif mo < ew:
        k1 = 0.424 * (1.0 - ((100.0 - rh_c) / 100.0) ** 1.7) + 0.0694 * math.sqrt(
            max(wind, 0.0)
        ) * (1.0 - ((100.0 - rh_c) / 100.0) ** 8)
        kw = k1 * 0.581 * math.exp(0.0365 * temp)
        m = ew - (ew - mo) * 10.0 ** (-kw)
    else:
        m = mo

    m = min(max(m, 0.0), 250.0)
    if m >= 250.0:
        return 101.0
    return 59.5 * (250.0 - m) / (147.2 + m)


def _dmc(temp: float, rh: float, rain: float, dmc_prev: float, month: int) -> float:
    """Duff Moisture Code for one day (Van Wagner 1987, Eqs. 11-16)."""
    if rain > 1.5:
        re = 0.92 * rain - 1.27
        mo = 20.0 + math.exp(5.6348 - dmc_prev / 43.43)
        if dmc_prev <= 33.0:
            b = 100.0 / (0.5 + 0.3 * dmc_prev)
        elif dmc_prev <= 65.0:
            b = 14.0 - 1.3 * math.log(dmc_prev)
        else:
            b = 6.2 * math.log(dmc_prev) - 17.2
        mr = mo + 1000.0 * re / (48.77 + b * re)
        dmc_prev = max(244.72 - 43.43 * math.log(max(mr - 20.0, 1e-6)), 0.0)

    t = max(temp, -1.1)
    le = _DMC_DAY_LENGTH[min(max(month, 1), 12) - 1]
    # Eq. 16: log drying rate; daily increment is added directly to the code.
    k = 1.894 * (t + 1.1) * (100.0 - min(max(rh, 0.0), 100.0)) * le * 1e-4
    return max(dmc_prev + k, 0.0)


def _dc(temp: float, rain: float, dc_prev: float, month: int) -> float:
    """Drought Code for one day."""
    if rain > 2.8:
        rd = 0.83 * rain - 1.27
        qo = 800.0 * math.exp(-dc_prev / 400.0)
        qr = qo + 3.937 * rd
        dc_prev = 400.0 * math.log(max(800.0 / qr, 1e-6))

    t = max(temp, -2.8)
    lf = _DC_DAY_LENGTH[min(max(month, 1), 12) - 1]
    v = max(0.36 * (t + 2.8) + lf, 0.0)
    return max(dc_prev + 0.5 * v, 0.0)


def _isi(wind: float, ffmc: float) -> float:
    """Initial Spread Index."""
    m = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    fw = math.exp(0.05039 * max(wind, 0.0))
    ff = 91.9 * math.exp(-0.1386 * m) * (1.0 + m ** 5.31 / 4.93e7)
    return 0.208 * fw * ff


def _bui(dmc: float, dc: float) -> float:
    """Buildup Index."""
    if dmc <= 0.0 or dc <= 0.0:
        return 0.0
    if dmc <= 0.4 * dc:
        return 0.8 * dmc * dc / (dmc + 0.4 * dc)
    # Van Wagner (1987) formulation; at very small dmc/dc the expression can
    # go slightly negative — the BUI is non-negative by definition, so clamp.
    return max(
        dmc - (1.0 - 0.8 * dc / (dmc + 0.4 * dc)) * (0.92 + (0.0114 * dmc) ** 1.7),
        0.0,
    )


def _fwi(isi: float, bui: float) -> float:
    """Fire Weather Index."""
    if bui <= 80.0:
        fd = 0.626 * bui ** 0.809 + 2.0
    else:
        fd = 1000.0 / (25.0 + 108.64 * math.exp(-0.023 * bui))
    b = 0.1 * isi * fd
    if b <= 1.0:
        return b
    return math.exp(2.72 * (0.434 * math.log(b)) ** 0.647)


def _dsr(fwi: float) -> float:
    """Daily Severity Rating."""
    return 0.0272 * fwi ** 1.77


def danger_class(fwi: float, simple: bool = True) -> str:
    """Map an FWI value to a danger class label."""
    classes = SIMPLE_CLASSES if simple else EFFIS_CLASSES
    for threshold, label in classes:
        if fwi < threshold:
            return label
    return classes[-1][1]


def compute_daily_fwi(
    temp_c: float,
    rh_pct: float,
    wind_kmh: float,
    rain_mm: float,
    month: int,
    ffmc_prev: float = DEFAULT_SEED["ffmc"],
    dmc_prev: float = DEFAULT_SEED["dmc"],
    dc_prev: float = DEFAULT_SEED["dc"],
    date: str = "",
) -> FWIDay:
    """
    Compute the full FWI System for one day.

    Parameters
    ----------
    temp_c : float
        Noon (or daily-max screening proxy) temperature in degC.
    rh_pct : float
        Noon (or daily-min screening proxy) relative humidity in percent.
    wind_kmh : float
        10 m wind speed in km/h.
    rain_mm : float
        24 h precipitation in mm.
    month : int
        Month (1-12), used for the day-length tables.
    ffmc_prev, dmc_prev, dc_prev : float
        Previous day's codes (carry-over state).
    date : str
        Optional ISO date label.
    """
    ffmc = _ffmc(temp_c, rh_pct, wind_kmh, rain_mm, ffmc_prev)
    dmc = _dmc(temp_c, rh_pct, rain_mm, dmc_prev, month)
    dc = _dc(temp_c, rain_mm, dc_prev, month)
    isi = _isi(wind_kmh, ffmc)
    bui = _bui(dmc, dc)
    fwi = _fwi(isi, bui)
    return FWIDay(
        date=date,
        ffmc=ffmc,
        dmc=dmc,
        dc=dc,
        isi=isi,
        bui=bui,
        fwi=fwi,
        dsr=_dsr(fwi),
        danger_class=danger_class(fwi),
    )


def compute_fwi_series(
    days: Sequence[Dict[str, float]],
    seed: Optional[Dict[str, float]] = None,
) -> List[FWIDay]:
    """
    Compute the FWI System over a consecutive day series.

    Parameters
    ----------
    days : sequence of dicts
        Each dict provides ``date`` (ISO str), ``temp_c``, ``rh_pct``,
        ``wind_kmh``, ``rain_mm``. The series must be consecutive and ordered;
        the codes carry state from day to day.
    seed : dict, optional
        Startup codes; defaults to the conventional FFMC=85, DMC=6, DC=15.

    Returns
    -------
    List[FWIDay]
        One FWIDay per input day.
    """
    state = dict(seed or DEFAULT_SEED)
    out: List[FWIDay] = []
    for day in days:
        month = int(str(day.get("date", "2026-06-01"))[5:7])
        result = compute_daily_fwi(
            temp_c=float(day["temp_c"]),
            rh_pct=float(day["rh_pct"]),
            wind_kmh=float(day["wind_kmh"]),
            rain_mm=float(day.get("rain_mm", 0.0)),
            month=month,
            ffmc_prev=state["ffmc"],
            dmc_prev=state["dmc"],
            dc_prev=state["dc"],
            date=str(day.get("date", "")),
        )
        state = {"ffmc": result.ffmc, "dmc": result.dmc, "dc": result.dc}
        out.append(result)
    return out
