# TX Engine (tx_core) — the single analytical authority

**Status:** Phase 1 — standalone package wired into the live platform (strangler
pattern). The website keeps running on the existing `src/` modules exactly as
before; TX Core is an additive orchestrator over them.

## 1. What TX Core is

`tx_core` is the web-free Python package that every Talaix surface (website,
REST API, SDK, CLI, QGIS plugin) will eventually consume for analysis. It is
the engine layer of the TX vision (see `../TXEng.md`): one uniform analysis
contract, one registry, one reproducibility stamp — with **zero duplicated
analysis logic**.

**The strangler guarantee:** TX Core never re-implements a hazard, an
evidence record, or a model. It *orchestrates* the platform's existing wired
modules (`src/climate`, `src/prediction`, `src/gis_mapping`) through narrow
adapters (`tx_core/adapters/`). Existing behaviour is preserved by
construction — the site cannot break from TX Core, because TX Core adds new
surface (`/api/tx/*`, the `tx` CLI) and delegates to the same code the site
already runs.

## 2. Layout

```
tx_core/
├── __init__.py          public API (TXEngine, TxResult, …)
├── _version.py          single source of version stamps
├── models.py            standard TX envelope (dependency-free dataclasses)
├── engine.py            TXEngine orchestrator + TX analysis levels
├── jobs.py              standard Job Object (TxJob + store + runner)
├── registry.py          TX Registry facade over config/*.json
├── reporting.py         JSON / GeoJSON / markdown rendering
├── cli.py               `tx` CLI (argparse only, no extra deps)
├── __main__.py          `python -m tx_core`
└── adapters/            the ONLY place tx_core touches src.*
    ├── climate.py       src.climate (hazards, evidence, ontology, TAM)
    ├── products.py      src.climate product engines (insurance,
    │                    verification, sustainability, licensing) as TX-2 analyses
    ├── prediction.py    src.prediction (FWI, risk model)   [reserved]
    └── gis.py           src.gis_mapping (indices, landcover) [reserved]
```

Import-light: `import tx_core` pulls no heavy/network dependencies. Every
platform module is imported lazily inside the adapters.

## 3. The standard envelope (TxResult)

Every TX analysis returns one uniform envelope (see `tx_core/models.py`):

| Key | Meaning |
| --- | --- |
| `analysis_id` | deterministic `TX-YYYYMMDD-<hex8>` (same inputs+day → same id) |
| `location` | `{lat, lon, name}` |
| `depth` | `quick` \| `standard` \| `deep` |
| `status` | `ok` \| `partial` \| `unavailable` |
| `results[]` | one `TxHazardResult` per hazard (status, level, blocks, evidence, provenance) **or per requested product analysis**, each stamped `tx_level` |
| `status_counts` | aggregate of per-hazard statuses |
| `evidence[]` | flattened evidence records (platform `EvidenceRecord` shape) |
| `sources[]` | de-duplicated official data sources |
| `engine_version` / `tx_version` / `tam_version` | reproducibility stamps |
| `generated_at` | UTC ISO-8601 (`Z`) — the platform's single clock |

Honesty contract (unchanged from `docs/EVIDENCE_ARCHITECTURE.md`): a hazard
that cannot produce real data returns `status="unavailable"` with an explicit
`unavailable_reason` — never a fabricated number. Unknown hazard ids are
either dropped (when requesting) or reported unavailable (when registered but
unresolvable).

## 4. Analysis levels

Advertised TX layers (used progressively; never faked):

| Level | Name | Phase-1 status |
| --- | --- | --- |
| TX-0 | Retrieval | platform data retrieval (implicit) |
| TX-1 | Deterministic | hazard screening modules (`src/climate/hazards/*`) |
| TX-2 | Statistical | product engines (`insurance`, `verification`, `sustainability`, `licensing`) registered as location-first TX analyses (`adapters/products.py`) — they run only on explicit request (`analyses=[...]`), land in the same `results[]` stamped `tx_level=2`, and never change a hazard-only `analysis_id`. The `insurance` product embeds the actuarial layer (`src/climate/actuarial.py`): exact Poisson frequency intervals, exceedance probabilities, return periods, severity statistics, collective-risk moments and the EN/AR actuarial reference — all inside `results[].blocks`, so every TX consumer (API, CLI, SDK, QGIS) receives it unchanged |
| TX-3 | Spatial | reserved (GIS indices / grids) |
| TX-4 | Predictive | reserved |
| TX-5 | ML | reserved (trained models) |
| TX-6 | Research | reserved |
| TX-7 | Reasoning | reserved (AI synthesis over TX results only) |
| TX-8 | Decision Intelligence | reserved |

**Honest scope note:** only engines with a real public *location-first*
entry point are registered as TX analyses. Claim-first engines (forensics
`assess_case`, supply-chain `evaluate_claim`) require a case/claim request
axis that `(lat, lon)` cannot honestly provide — registering them as
location analyses would fake a capability. They remain available through
their own v2 endpoints; a future TX case/claim axis can register them
without pretence.

## 5. Reproducibility

- Every result stamps `engine_version` / `tx_version` / `tam_version`.
- `analysis_id` is deterministic per (location, hazards, depth, versions, UTC
  day) — re-running the same inputs on the same day yields the same id.
- The TX Registry (`tx_core/registry.py`, backed by `config/model_registry.json`,
  `config/source_registry.json`, `config/data_registry.json`) is the lookup
  table: each analysis can state exactly which model version, datasets and
  sources it rests on.
- Future: `tx reproduce <analysis_id>` replays an analysis deterministically.

## 6. CLI

```
python -m tx_core version          # engine versions
python -m tx_core registry --json  # registry digest
python -m tx_core hazards          # registered hazards
python -m tx_core sources          # official data sources
python -m tx_core analyze --lat 41.5 --lon -8.6 --hazard wildfire [--json] [--format md]
python -m tx_core reproduce result.json [--json]   # replay + verify a saved result
```

`tx reproduce` reads a saved TxResult envelope (the output of
`tx analyze --json` or a `/api/tx/jobs/<id>/result` body), re-runs the
recorded request (location, hazards, depth) through the engine, and
verifies reproducibility honestly: per-hazard status equality is the
substantive check; `analysis_id` equality is reported separately because
the id is day-scoped. Exit codes: 0 reproduced, 1 diverged, 2 usage error.

Installed package also exposes the `tx` console entry point
(`pyproject.toml [project.scripts]`).

## 7. Web wiring (/api/tx) — additive only

`src/dashboard/tx_api.py` registers a Flask blueprint at `/api/tx` inside the
existing `create_app()` (next to the v2 blueprints). No existing route is
modified:

```
GET /api/tx/health     engine + registry availability
GET /api/tx/version    engine versions + TX levels
GET /api/tx/hazards    registered hazard descriptors
GET /api/tx/sources    official sources
GET /api/tx/registry   registry digest
GET /api/tx/analyze    ?lat=&lon=[&hazard=..][&analysis=..][&depth=][&name=]
POST /api/tx/run       submit a job {lat, lon, [hazards], [analyses], [depth], [name]}
GET /api/tx/jobs/<id>  poll job status + progress
GET /api/tx/jobs/<id>/result  fetch the TxResult envelope (when succeeded)
GET /api/tx/products   registered TX product engines (TX-2+ analyses)
```

The "site not broken" guarantee is enforced by tests
(`tests/test_tx_core.py::test_existing_routes_untouched`) asserting
pre-existing routes (`/api/status`, `/api/health`) still serve 200 through the
same app instance.

### 7.1 TX-0/TX-1 facade over the legacy v1 endpoint (`GET /api/analyze`)

The real `/api/analyze` pipeline now runs **behind TXEngine**:

```
GET /api/analyze
  -> api.py: _tx_engine().legacy_analyze(lat, lon, name)      (TX facade)
  -> tx_core.engine.TXEngine.legacy_analyze()                 (validate + orchestrate)
  -> tx_core.adapters.legacy_v1.cached_analysis()             (the ONLY tx_core touchpoint)
  -> src.dashboard.snapshot.cached_analysis()                 (unchanged, 15-min TTL)
  -> TalaixRealAnalyser.analyse_point()                       (unchanged real pipeline)
```

`legacy_analyze()` returns `(payload, tx_meta)`:

- **`payload`** — the EXACT legacy v1 dict (same analyser, same cache, same
  error mapping). The wire contract is byte-identical: a test proves the
  response equals a control route that jsonifies the engine's payload
  (`test_api.py::test_analyze_flows_through_tx_engine_byte_identical`).
- **`tx_meta`** — a separate side-channel (deterministic `analysis_id`,
  engine/tx/tam versions, depth, `generated_at`) never injected into the
  legacy payload; available for audit/telemetry and future `tx reproduce`.

The route's factory `src.dashboard.api._tx_engine()` is lazy (tx_core imports
nothing heavy at module import) and is monkeypatched in tests to stay
network-free. Other consumers of the same cached payload (`/api/report`,
`/api/ignition-risk`, `/api/exposure-summary`, deprecated `POST /api/risk`)
are intentionally untouched — their migration is future work with their own
contract proofs.

### 7.2 Standard Job Object (`POST /api/tx/run` → poll → result)

Deep analyses can take minutes, so the async surface is job-based (the job
object shape is the stable contract — see `tx_core/jobs.py`):

```
POST /api/tx/run                   {lat, lon, [hazards], [analyses], [depth], [name]}
  -> 202 {job_id, status, poll, result_url, ...}   (new job accepted)
  -> 200 same payload                              (idempotent resubmission)
GET /api/tx/jobs/<job_id>          status + progress {completed, total}
GET /api/tx/jobs/<job_id>/result   the TxResult envelope when succeeded
                                   (409 with honest state/error otherwise)
```

Contract points:

- `job_id` is deterministic (`TXJ-YYYYMMDD-<hex8>` over the request +
  `tx_version` + UTC day): re-submitting the same analysis on the same day
  returns the existing job instead of re-running the pipeline.
- Lifecycle is `queued → running → succeeded | failed`. Failures carry the
  real error message; a result is never fabricated.
- Progress is per-hazard: `TXEngine.analyze(..., on_hazard=cb)` reports
  `(result, completed, total)` after each hazard; callback errors are
  swallowed so bookkeeping can never break an analysis.
- Phase-1 backend is in-process (`TxJobStore` thread-safe dict with TTL
  eviction + bounded `ThreadPoolExecutor` in `TxJobRunner`) — honest for a
  single-process deployment. The store/runner interfaces are the seam: a
  Redis/queue backend can replace them without touching routes or clients
  (required before multi-worker/multi-node deployments).
- The web layer keeps one module-level runner (`tx_api._JOB_RUNNER`);
  tests replace it with a synchronous runner over an injected fake engine
  (`tests/test_tx_jobs.py`).

## 8. Testing

```
.venv/bin/python -m pytest tests/test_tx_core.py -q
```

Network-free: the engine is tested with an injected fake hazard registry; the
blueprint is tested with a patched engine factory. A regression test also
exercises the **default (non-injected) adapter path** — a live CLI run caught
an `AttributeError` there (`adapters.hazard_ids()`, the adapter package only
re-exports submodules), fixed by importing `tx_core.adapters.climate`
explicitly, the same pattern as `tx_core.registry`. Run the broader regression
suite after changes touching `src/dashboard/api.py`:

```
.venv/bin/python -m pytest tests/test_api.py tests/test_climate_core.py -q
```

## 9. Client surfaces (SDK, QGIS) — same engine, every surface

Every client consumes the TX API contracts above — no client re-implements
analysis, and all clients inherit the honesty contract (unavailable/failed
states are data, never exceptions or fabrications).

**Python SDK** (`sdk/python/hydrashield/tx.py` — `TxClient`, stdlib-only,
exported from the `hydrashield` package):

```python
from hydrashield import TxClient

tx = TxClient()                                   # https://talaix.com
quick = tx.analyze(49.96, 6.03, hazards=["wildfire"], depth="quick")
job = tx.run(49.96, 6.03, depth="deep")           # POST /api/tx/run
result = tx.wait(job["job_id"], on_poll=print)    # poll → TxResult envelope
```

Introspection (`health/version/hazards/sources/registry`) mirrors the
REST surface; `run/job/result/wait` implement the standard Job Object.
Offline tests: `sdk/python/tests/test_tx.py` (re-exported into the main
suite by `tests/test_sdk_tx.py`).

**JavaScript SDK** (`sdk/js/hydrashield.js`): the same TX surface
(`txAnalyze`, `txRun`, `txJob`, `txResult`, `txWait` + introspection),
fetch-based, zero dependencies. Offline tests:
`node sdk/js/test_hydrashield.node.js`.

**QGIS plugin** (`qgis-plugin/hydrashield/`): the Processing algorithm
**"Analyze point (Talaix TX Engine)"** (`hydrashield:analyze_tx_point`)
consumes `GET /api/tx/analyze` and emits one feature per hazard result —
attributes carry the level, the honesty fields (status, basis, validated,
unavailable_reason) and the envelope stamps (analysis_id, depth,
engine_version). URL building and TxResult normalization are pure
functions (`tx_client.py`) unit-tested outside QGIS
(`tests/test_qgis_tx_client.py`); the network runs through
QgsNetworkAccessManager on the Processing worker thread, per official
plugin guidance.

## 10. Roadmap (next phases)

1. [DONE] Move the `/api/analyze` real-data pipeline behind TX Engine
   (TX-0/TX-1 facade, §7.1 — `TXEngine.legacy_analyze()` + `adapters/legacy_v1.py`;
   contract proven byte-identical by tests).
2. [DONE] Standard Job Object: `POST /api/tx/run` → job_id → polling
   (`tx_core/jobs.py` + §7.2 — deterministic idempotent job ids, per-hazard
   progress, honest failures; in-process phase-1 backend behind replaceable
   store/runner interfaces).
3. [DONE] `talaix-sdk` (Python/JS) over the TX API; `tx reproduce`; QGIS
   plugin consuming `/api/tx/analyze` (§6, §9 — `TxClient`/`tx*` JS methods,
   `hydrashield:analyze_tx_point` Processing algorithm).
4. [DONE] Register product engines (insurance, forensics, sustainability,
   supply chain) as TX analyses (TX-2+) — done for the location-first
   engines (`insurance`, `verification` as the site-level face of the
   forensics stack, `sustainability`, `licensing` as the site-level
   licensing dossier) via `adapters/products.py` and the
   `analyses=[...]` request axis on every surface (API, jobs, CLI, SDKs,
   QGIS). Claim-first engines (forensics cases, supply-chain claims) are
   honestly NOT location analyses — see the scope note in §4.
