# Quickstart — reproduce a multi-hazard screening locally

Copy-paste recipe, from a fresh clone to a reproducible result.
Python 3.10–3.12 required.

```bash
git clone https://github.com/Motaz3d/tore.git
cd tore
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## 1. See what the engine knows

```bash
tx hazards      # ten registered hazard modules
tx sources      # the official open data sources behind them
tx registry     # models / datasets / sources digest
```

## 2. Analyse a location

```bash
tx analyze --lat 36.8 --lon 34.6 --name "Mersin, Türkiye"
```

Every hazard answers with a status: `ok` with a screening level, or
`unavailable` with the honest reason (missing dataset coverage, network,
etc.). The engine never invents a number.

Depths: `--depth quick|standard|deep` (default `standard`).
Single hazard: `--hazard wildfire` (repeatable).

## 3. Save the envelope and replay it later

```bash
tx analyze --lat 36.8 --lon 34.6 --json > mersin.json
tx reproduce mersin.json
```

`tx reproduce` re-runs the exact recorded request and reports whether the
result still holds (`REPRODUCED` / `DIVERGED` per hazard). The
`analysis_id` is day-scoped — identical inputs on a different UTC day get a
different id; status equality is the substantive check.

## 4. Use it from Python

```python
from tx_core import TXEngine
from tx_core.reporting import result_to_markdown, result_to_geojson

engine = TXEngine()
result = engine.analyze(lat=36.8, lon=34.6,
                        hazards=["wildfire", "flood", "heat"],
                        depth="standard", name="Mersin, Türkiye")

open("mersin.md", "w").write(result_to_markdown(result))
open("mersin.geojson", "w").write(result_to_geojson(result))
```

## Notes

- Hazard modules fetch **live open data** (Copernicus, NASA, USGS, …), so
  analyses need network access; without it, affected hazards answer
  `unavailable` with the reason — by design.
- Heavy EO reads (STAC/COG) need the `gis` extra: `pip install -e ".[gis]"`.
- The knowledge registries in `config/` are loaded relative to the working
  directory — run the CLI from the repository root (as here).
