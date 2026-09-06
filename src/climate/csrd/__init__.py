"""
CsrdTX — Talaix CSRD/ESRS regulatory intelligence engine.

A version-aware engine that answers four questions deterministically:

1. **Applicability** — is this company in scope of the CSRD, for which
   reporting year, under which rule set (and under the proposed Omnibus
   changes)?
2. **Double materiality** — which ESRS topics are material, scored with a
   documented, reproducible formula, seeded with real physical-risk
   evidence where Talaix has it.
3. **Coverage & gaps** — which disclosure requirements are covered by
   evidence, which are partial, which are declared gaps.
4. **Readiness** — a weighted 0–100 readiness score with an itemised gap
   analysis.

Design norms (docs/CSRD_TX_ENGINE.md):

- **Rules are data, not code.** Regulatory knowledge lives in
  ``config/csrd/*.json`` with legal status and sources. A regulatory
  update is a data change, never an engine rewrite.
- **Never invent.** Every datapoint carries a controlled status
  (VERIFIED / SUPPORTED / COMPANY_DECLARED / INFERRED / UNAVAILABLE /
  NOT_ASSESSED) and a reason. Missing data is declared, not fabricated.
- **Deterministic.** Same input → same scores → same content hash. No
  randomness, no hidden state, no network calls in scoring code.
- **Screening, not assurance.** Output supports CSRD/ESRS preparation;
  it is not legal advice, not assurance under ISAE 3000, and not the
  limited-assurance engagement CSRD requires from an auditor.

No Flask imports in this package; the API layer is ``api_csrd.py``.
"""

from .applicability import assess_applicability
from .engine import ENGINE_VERSION, build_csrd_assessment
from .materiality import assess_topic, score_financial, score_impact
from .readiness import build_gap_analysis, compute_readiness
from .regulations import esrs_version, esrs_versions, load_changelog, rule_set_for_year

__all__ = [
    "ENGINE_VERSION",
    "assess_applicability",
    "assess_topic",
    "build_csrd_assessment",
    "build_gap_analysis",
    "compute_readiness",
    "esrs_version",
    "esrs_versions",
    "load_changelog",
    "rule_set_for_year",
    "score_financial",
    "score_impact",
]
