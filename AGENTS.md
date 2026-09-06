# AGENTS.md — tore (Talaix Open Risk Engine)

## What this repository is

The **public open-source mirror of the Talaix analytical engine**, published
for the NLnet Restack application and the open-source community. It contains
only the analytical core: `tx_core`, `src/climate` (minus `api_*.py` web
blueprints), `src/prediction`, `src/gis_mapping`, the analytical
data-pipeline modules of `src/dashboard`, `src/hydration_control`, plus the
`config/` knowledge registries, `docs/TX_ENGINE.md`, and a network-free
test subset.

## Binding rule (read before editing)

**Development happens in the platform monorepo** (`HydraShield/hydra-shield-platform`).
Do not edit engine code directly in this repository — changes here would be
overwritten by the next sync. The correct flow:

1. Change the engine in the monorepo.
2. Run `hydra-shield-platform/scripts/sync_tore.sh --push -m "<message>"`.
3. That script copies the exact mirrored set and pushes to GitHub.

Repository-local files (never touched by the sync): `README.md`, `LICENSE`,
`CHANGELOG.md`, `pyproject.toml`, `.gitignore`, `.github/`, `examples/`,
`tests/fakes.py`, `tests/test_engine_smoke.py`, this `AGENTS.md`.

## Commands

- Install: `pip install -e ".[dev]"` (extras: `gis`, `ml`)
- Tests: `python -m pytest tests/ -v` (must stay green standalone)
- CLI: `tx analyze --lat <lat> --lon <lon>` from the repo root
- Verify no web-layer leakage: no module here may import
  `src.dashboard.api`, `src.dashboard.accounts`, billing, marketing or SMS.

## Honesty contract

Screening indicators only; `unavailable` with an explicit reason is a
first-class result; no fabricated numbers; provenance on every output;
`tx reproduce` must be able to replay any saved envelope.
