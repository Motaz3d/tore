"""
Eurostat construction-cost index (STS_COPI_A) — official price calibration
for the Talaix loss screening estimate (docs/LOSS_DATA_ACQUISITION.md §2).

Fetches the annual Eurostat series "Construction producer prices or costs,
new residential buildings" (index 2015=100) via the SDMX 2.1 TSV endpoint,
parses the COST rows per country, and derives the calibration factor
between the benchmarks' declared price-basis year and the latest official
index value. The whole table is fetched once and cached (7 days — the
series is annual).

Honesty contract: the factor scales all three declared bands equally —
the official index DATES the benchmarks, it never narrows them. Every
output carries the source, both index values, both years and the Eurostat
flags (p = provisional, e = estimated, i = value with metadata).
"""

from __future__ import annotations

import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from ..dashboard.cache import cached

CCI_URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/"
    "STS_COPI_A?format=TSV"
)

CCI_SOURCE = (
    "Eurostat STS_COPI_A — Construction producer prices or costs, new "
    "residential buildings, annual (2015=100)"
)

TTL_CCI = 7 * 24 * 3600.0  # 7 days — the series is annual


@cached("eurostat_cci_tsv", TTL_CCI)
def _fetch_cci_tsv() -> str:
    """Download the STS_COPI_A TSV (all countries, one fetch). Cached 7 d."""
    req = urllib.request.Request(
        CCI_URL, headers={"User-Agent": "Talaix-LossCalibration/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=40.0) as resp:
            return resp.read().decode("utf-8")
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"Eurostat STS_COPI_A fetch failed: {exc}") from exc


def parse_cci_tsv(text: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Parse the SDMX TSV into {geo: {year: {"value", "flag"}}}.

    The dataset mixes two series kinds: ``COST`` (construction cost) and
    ``PRC_PRR`` (producer price) — some countries (e.g. Luxembourg) only
    publish the price series. Both index construction prices/costs, so a
    country is served by its COST series when present, else its PRC_PRR
    series; the chosen series is tagged on each point for the method note.
    The ``I15`` unit (2015=100) is preferred when several index bases
    exist — the calibration uses ratios, so the base year cancels out.

    Cells look like ``"134.4 p"`` (provisional), ``"86.2 i"`` (metadata
    flag) or ``":"`` (missing). The flag letters are preserved.
    """
    # First pass: {geo: {(indic_bt, unit): {year: {...}}}}.
    raw: Dict[str, Dict[Any, Dict[int, Dict[str, Any]]]] = {}
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return {}
    header = lines[0].split("\t")
    years = []
    for cell in header[1:]:
        cell = cell.strip()
        years.append(int(cell) if cell.isdigit() else None)

    for line in lines[1:]:
        cells = line.split("\t")
        dims = cells[0].split(",")
        if len(dims) < 6 or dims[1] not in ("COST", "PRC_PRR"):
            continue
        indic_bt, unit, geo = dims[1], dims[4], dims[5].strip()
        series: Dict[int, Dict[str, Any]] = {}
        for idx, year in enumerate(years):
            if year is None or idx + 1 >= len(cells):
                continue
            cell = cells[idx + 1].strip()
            if not cell or cell == ":":
                continue
            parts = cell.split()
            try:
                value = float(parts[0])
            except (ValueError, IndexError):
                continue
            series[year] = {"value": value, "flag": " ".join(parts[1:]).strip()}
        if series:
            raw.setdefault(geo, {})[(indic_bt, unit)] = series

    # Second pass: pick the best series per country and tag it.
    out: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for geo, variants in raw.items():
        chosen_key = None
        for prefer_indic in ("COST", "PRC_PRR"):
            if (prefer_indic, "I15") in variants:
                chosen_key = (prefer_indic, "I15")
                break
            for key in variants:
                if key[0] == prefer_indic:
                    chosen_key = key
                    break
            if chosen_key:
                break
        if not chosen_key:
            continue
        series = dict(variants[chosen_key])
        for point in series.values():
            point["indic_bt"] = chosen_key[0]
            point["unit"] = chosen_key[1]
        out[geo] = series
    return out


def latest_cci(geo: str) -> Optional[Dict[str, Any]]:
    """The latest official index point for a country, or None."""
    try:
        table = parse_cci_tsv(_fetch_cci_tsv())
    except RuntimeError:
        return None
    series = table.get(geo) or {}
    if not series:
        return None
    year = max(series)
    return {"year": year, "value": series[year]["value"],
            "flag": series[year]["flag"],
            "indic_bt": series[year].get("indic_bt"),
            "unit": series[year].get("unit")}


def calibration(geo: Optional[str], basis_year: int = 2023) -> Dict[str, Any]:
    """Calibration factor between the benchmarks' price-basis year and the
    latest official Eurostat index for ``geo``.

    factor = CCI(latest) / CCI(basis_year) — applied to all declared bands
    equally. ``status: unavailable`` (with reason) when the index cannot be
    fetched or the country/years are missing; callers then keep the
    declared bands unchanged.
    """
    if not geo:
        return {"status": "unavailable",
                "reason": "no country benchmark matched — generic defaults in use"}
    try:
        table = parse_cci_tsv(_fetch_cci_tsv())
    except RuntimeError as exc:
        return {"status": "unavailable", "reason": str(exc)}
    series = table.get(geo) or {}
    if not series:
        return {"status": "unavailable",
                "reason": f"Eurostat STS_COPI_A has no construction cost/price "
                          f"series for {geo}"}
    latest_year = max(series)
    latest = series[latest_year]
    basis = series.get(basis_year)
    basis_year_used = basis_year
    basis_note = ""
    if not basis:
        # Several countries publish the annual series with a lag (e.g. the
        # series stops at 2022): use the nearest earlier official value and
        # say so — never a silent substitution.
        earlier = [y for y in series if y <= basis_year]
        if not earlier:
            return {"status": "unavailable",
                    "reason": f"Eurostat STS_COPI_A {geo}: no index value at "
                              f"or before {basis_year}"}
        basis_year_used = max(earlier)
        basis = series[basis_year_used]
        basis_note = (f"Requested basis year {basis_year} is not published "
                      f"for {geo}; the nearest earlier official value "
                      f"({basis_year_used}) is used.")
    factor = latest["value"] / basis["value"]
    series_kind = latest.get("indic_bt") or basis.get("indic_bt")
    return {
        "status": "ok",
        "factor": round(factor, 4),
        "basis_year": basis_year,
        "basis_year_used": basis_year_used,
        "basis_value": basis["value"],
        "latest_year": latest_year,
        "latest_value": latest["value"],
        "flags": {"latest": latest["flag"], "basis": basis["flag"]},
        "series": {"indic_bt": series_kind,
                   "unit": latest.get("unit") or basis.get("unit")},
        "source": CCI_SOURCE,
        "url": CCI_URL,
        "method": (
            f"All declared benchmark bands are scaled by the official index "
            f"ratio {latest['value']} ({latest_year}) / {basis['value']} "
            f"({basis_year_used}) = {round(factor, 4)}"
            + (f" [{series_kind} series]" if series_kind else "")
            + (f" {basis_note}" if basis_note else "")
            + " The index dates the benchmarks; band widths are unchanged."),
    }
