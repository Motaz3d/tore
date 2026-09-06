# Changelog

All notable changes to the Talaix Open Risk Engine are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to the engine contract in `docs/TX_ENGINE.md`.

## [0.1.0] — 2026-09-06

Initial public release: the analytical engine extracted from the Talaix
platform monorepo and published as open source (EUPL-1.2).

- `tx_core` — the TX engine: stable contract (`TXEngine.analyze`),
  `TxResult` envelope with evidence + provenance, Markdown/GeoJSON
  renderers, async job runner, and the `tx` CLI
  (`analyze` / `hazards` / `products` / `sources` / `registry` /
  `reproduce` / `version`).
- `src.climate` — multi-hazard core: ontology, evidence architecture,
  hazard registry with ten hazard modules (wildfire, flood, drought, heat,
  wind, coastal, cyclone, dust, volcanic, earthquake), product engines
  (insurance, licensing, verification, sustainability) and CSRD/ESRS E1
  support.
- `src.prediction` — Canadian FWI fire-danger system, wildfire risk ML,
  spread screening.
- `src.gis_mapping` — Earth-observation ingestion: spectral indices
  (NDVI/NDMI/NDWI), ESA WorldCover land cover.
- `config/` — versioned knowledge registries consumed by the engine.
- Honesty contract enforced end-to-end: unavailable hazards are reported
  with explicit reasons; no fabricated figures; reproducible envelopes
  verifiable via `tx reproduce`.
