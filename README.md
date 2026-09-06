# Talaix Open Risk Engine (`tore`)

An open, reproducible analytical engine that turns open Earth-observation
and climate data into locally computable **multi-hazard risk indicators** —
with evidence, provenance and engine versions attached to every result.

`tore` is the analytical core behind [Talaix](https://talaix.com), published
as open source so that researchers, civil-protection teams, journalists,
and developers can run the same analysis locally, audit it, and build on it.

Ten hazard modules — **wildfire, flood, drought, extreme heat, extreme wind,
coastal exposure, tropical cyclones, dust storms, volcanic activity and
earthquakes** — plus composite product analyses (insurance screening,
sustainability/CSRD physical-risk evidence, site verification, environmental
licensing screening).

## The honesty contract

- **Screening indicators, not certified predictions.** Every output is a
  screening-level indicator, labelled as such.
- **"Unavailable" is a first-class result.** A hazard that cannot produce
  real data is reported as `status="unavailable"` with an explicit reason —
  never filled in with invented numbers.
- **Provenance on everything.** Each result carries its evidence records,
  data sources and engine versions, so any analysis can be replayed and
  audited (`tx reproduce`).

## Install

Python 3.10–3.12. From this repository:

```bash
git clone https://github.com/Motaz3d/tore.git
cd tore
pip install -e .
```

Optional extras:

```bash
pip install -e ".[gis]"   # rasterio / pystac-client / pyproj / Pillow (heavy EO reads)
pip install -e ".[ml]"    # xgboost / joblib (ML risk models)
pip install -e ".[dev]"   # pytest
```

## Quick start (CLI)

```bash
# Analyse a location across all ten hazards
tx analyze --lat 39.0 --lon 35.0

# Pick hazards, depth and output format
tx analyze --lat 39.0 --lon 35.0 --hazard wildfire --hazard heat \
    --depth deep --format md

# Machine-readable envelope (JSON, with evidence + provenance)
tx analyze --lat 39.0 --lon 35.0 --json > result.json

# Replay a saved result and verify reproducibility
tx reproduce result.json

# List registered hazards, product engines and official data sources
tx hazards
tx products
tx sources
tx registry
tx version
```

Analysis depths: `quick` (fast screening), `standard` (default),
`deep` (full evidence chain).

## Quick start (Python)

```python
from tx_core import TXEngine

engine = TXEngine()
result = engine.analyze(lat=39.0, lon=35.0, hazards=["wildfire", "flood"],
                        depth="standard", name="Anatolia plateau")

print(result.status)                 # ok / partial / unavailable
for hazard in result.results:
    print(hazard.hazard, hazard.status, hazard.summary)

envelope = result.to_dict()          # JSON-serialisable, with provenance
```

Reporters: `tx_core.reporting.result_to_markdown(result)` and
`result_to_geojson(result)` render the same envelope as Markdown / GeoJSON.

## Repository layout

```
tore/
├── tx_core/            # the TX engine: stable contract, orchestration, CLI
│   ├── engine.py       #   TXEngine — the single entry point
│   ├── models.py       #   TxRequest / TxResult / TxHazardResult envelopes
│   ├── reporting.py    #   Markdown / GeoJSON renderers
│   ├── jobs.py         #   async job runner + store
│   └── adapters/       #   narrow, lazy adapters over the analytical modules
├── src/
│   ├── climate/        # multi-hazard core: ontology, evidence, registry,
│   │   ├── hazards/    #   ten hazard modules, product engines, CSRD
│   │   └── csrd/       #   ESRS E1 physical-risk support
│   ├── prediction/     # Canadian FWI, wildfire risk ML, spread screening
│   ├── gis_mapping/    # EO ingestion: Sentinel-2/Landsat, indices, land cover
│   ├── dashboard/      # analytical data pipeline only: real_data, exposure,
│   │                   #   real_analysis, snapshot, cache, … (no web app)
│   └── hydration_control/  # water-resource optimisation behind wildfire
├── config/             # versioned knowledge registries (JSON)
├── docs/TX_ENGINE.md   # the engine contract (TX levels, envelope, rules)
├── examples/           # reproducible recipes
└── tests/              # network-free engine + core tests
```

The layout mirrors the Talaix platform monorepo on purpose: `tx_core` never
re-implements analysis — it orchestrates the analytical modules through the
adapters, so the engine you run here is byte-identical to the one behind the
live platform. The web layer itself (Flask app, auth, billing, HTTP
blueprints) is **not** part of this repository — only the analytical core is.

## Data sources

Open data in, open data out: Copernicus ERA5/ERA5-Land, Sentinel-2 (STAC),
GloFAS & GEOGLOWS, NASA FIRMS & EONET, GDACS, USGS (ComCat, gauges),
EMSC, IBTrACS, ESA WorldCover, WorldPop, OpenStreetMap, NOAA NCEI and more.
Run `tx sources` for the registry, or browse `src/climate/data_registry.py`.

## Development and mirroring

Engine development happens in the Talaix platform monorepo; **this
repository is the canonical public mirror of the analytical engine**. Every
change to `tx_core/`, `src/climate/`, `src/prediction/` or `src/gis_mapping/`
is mirrored here. Issues and discussions are welcome on this repo.

## License

- Code: **EUPL-1.2** (see `LICENSE`)
- Knowledge registries / sample data (`config/`): **CC-BY-4.0**
- Documentation (`docs/`, `examples/`): **CC-BY-SA-4.0**

## Status

Early development (v0.1.0). This repository supports an application to the
[NLnet](https://nlnet.nl) Restack programme (Open Internet Stack).
