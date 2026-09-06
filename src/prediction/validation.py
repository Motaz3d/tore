"""
Validation foundation for Talaix risk products (scientific layer).

This module provides the metric machinery needed to validate Talaix
risk scores against *observed* fire events (NASA FIRMS detections):

    real scores + real labels
        -> confusion matrix (TP / FP / TN / FN)
        -> precision, recall, F1, accuracy, critical success index
        -> calibration / reliability bins + Brier score
        -> temporally separated train/evaluation splits (no leakage)
        -> a self-describing ValidationReport (period, coverage, sources,
           model version, assumptions, limitations, metrics)

It contains no data fabrication: it only computes statistics over samples
supplied by the caller. The real-data orchestration lives in
``scripts/run_validation.py``. Until that pipeline has been executed on
real historical data, no Talaix product may be described as validated.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple


# --------------------------------------------------------------------------
# Confusion matrix & scalar metrics
# --------------------------------------------------------------------------

@dataclass
class ConfusionMatrix:
    """Binary confusion matrix (positive class = observed fire)."""

    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @staticmethod
    def _safe_div(num: float, den: float) -> Optional[float]:
        return round(num / den, 4) if den else None

    @property
    def precision(self) -> Optional[float]:
        """TP / (TP + FP); None when nothing was predicted positive."""
        return self._safe_div(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> Optional[float]:
        """TP / (TP + FN); None when there were no actual positives."""
        return self._safe_div(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> Optional[float]:
        p, r = self.precision, self.recall
        if p is None or r is None or (p + r) == 0:
            return None
        return round(2.0 * p * r / (p + r), 4)

    @property
    def accuracy(self) -> Optional[float]:
        return self._safe_div(self.tp + self.tn, self.total)

    @property
    def critical_success_index(self) -> Optional[float]:
        """TP / (TP + FP + FN) — the threat score used in fire science."""
        return self._safe_div(self.tp, self.tp + self.fp + self.fn)

    @property
    def false_alarm_ratio(self) -> Optional[float]:
        """FP / (TP + FP)."""
        return self._safe_div(self.fp, self.tp + self.fp)

    def to_dict(self) -> Dict:
        return {
            "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
            "total": self.total,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "critical_success_index": self.critical_success_index,
            "false_alarm_ratio": self.false_alarm_ratio,
        }


def compute_confusion_matrix(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> ConfusionMatrix:
    """Count TP/FP/TN/FN over aligned boolean/0-1 sequences."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    cm = ConfusionMatrix()
    for t, p in zip(y_true, y_pred):
        t, p = bool(t), bool(p)
        if t and p:
            cm.tp += 1
        elif not t and p:
            cm.fp += 1
        elif not t and not p:
            cm.tn += 1
        else:
            cm.fn += 1
    return cm


# --------------------------------------------------------------------------
# Calibration / reliability
# --------------------------------------------------------------------------

def compute_calibration(
    scores: Sequence[float],
    y_true: Sequence[int],
    n_bins: int = 10,
    score_min: float = 0.0,
    score_max: float = 100.0,
) -> List[Dict]:
    """
    Reliability bins: mean predicted score vs observed fire frequency.

    Returns one entry per non-empty bin: range, count, mean predicted score
    and observed frequency. Empty bins are omitted (they carry no evidence).
    """
    if len(scores) != len(y_true):
        raise ValueError("scores and y_true must have equal length")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    width = (score_max - score_min) / n_bins
    bins: List[List[Tuple[float, int]]] = [[] for _ in range(n_bins)]
    for s, t in zip(scores, y_true):
        idx = int((float(s) - score_min) / width) if width else 0
        idx = max(0, min(n_bins - 1, idx))
        bins[idx].append((float(s), int(bool(t))))

    out = []
    for i, members in enumerate(bins):
        if not members:
            continue
        out.append(
            {
                "lower": round(score_min + i * width, 4),
                "upper": round(score_min + (i + 1) * width, 4),
                "count": len(members),
                "mean_predicted": round(sum(m[0] for m in members) / len(members), 4),
                "observed_frequency": round(sum(m[1] for m in members) / len(members), 4),
            }
        )
    return out


def brier_score(scores: Sequence[float], y_true: Sequence[int],
                score_max: float = 100.0) -> Optional[float]:
    """Brier score of the normalised (0-1) scores; None for empty input."""
    if len(scores) != len(y_true):
        raise ValueError("scores and y_true must have equal length")
    if not scores:
        return None
    total = 0.0
    for s, t in zip(scores, y_true):
        p = max(0.0, min(1.0, float(s) / score_max))
        total += (p - float(bool(t))) ** 2
    return round(total / len(scores), 4)


def roc_auc(scores: Sequence[float], y_true: Sequence[int]) -> Optional[float]:
    """
    ROC-AUC via the rank-based Mann-Whitney U statistic (tie-corrected).

    Statistically appropriate only with both classes present; returns None
    otherwise (undefined), never a fabricated number. Threshold-free ranking
    quality of the score, not evidence of calibration.
    """
    if len(scores) != len(y_true):
        raise ValueError("scores and y_true must have equal length")
    pairs = sorted(zip(scores, (1 if bool(t) else 0 for t in y_true)), key=lambda p: p[0])
    n_pos = sum(t for _, t in pairs)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    # Average ranks with tie correction.
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    rank_sum_pos = sum(r for r, (_, t) in zip(ranks, pairs) if t == 1)
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return round(auc, 4)


def pr_auc(scores: Sequence[float], y_true: Sequence[int]) -> Optional[float]:
    """
    PR-AUC: area under the precision-recall curve (trapezoidal over recall).

    More informative than ROC-AUC under class imbalance (rare positives such
    as fire detections). Returns None when no positive observation exists.
    The no-skill baseline equals the positive class prevalence — always
    report it alongside.
    """
    if len(scores) != len(y_true):
        raise ValueError("scores and y_true must have equal length")
    pairs = sorted(zip(scores, (1 if bool(t) else 0 for t in y_true)),
                   key=lambda p: p[0], reverse=True)
    n_pos = sum(t for _, t in pairs)
    if n_pos == 0:
        return None
    tp = 0
    fp = 0
    prev_recall = 0.0
    prev_precision = 1.0
    area = 0.0
    for _s, t in pairs:
        if t:
            tp += 1
        else:
            fp += 1
        recall = tp / n_pos
        precision = tp / (tp + fp)
        area += (recall - prev_recall) * (precision + prev_precision) / 2.0
        prev_recall = recall
        prev_precision = precision
    return round(area, 4)


# --------------------------------------------------------------------------
# Temporal separation (anti-leakage)
# --------------------------------------------------------------------------

def temporal_train_test_split(
    dates: Sequence[str], train_fraction: float = 0.6
) -> Tuple[List[int], List[int]]:
    """
    Split sample indices into train / evaluation partitions by date.

    All train dates are strictly earlier than all evaluation dates, so a
    model (or a threshold) tuned on the train partition is never evaluated
    on observations it has seen. Raises ValueError when the split is
    degenerate (fewer than 2 distinct dates).
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    unique = sorted(set(dates))
    if len(unique) < 2:
        raise ValueError("Need at least 2 distinct dates for a temporal split")
    cut = max(1, min(len(unique) - 1, int(round(len(unique) * train_fraction))))
    train_dates = set(unique[:cut])
    train_idx = [i for i, d in enumerate(dates) if d in train_dates]
    test_idx = [i for i, d in enumerate(dates) if d not in train_dates]
    return train_idx, test_idx


def select_threshold(
    scores: Sequence[float], y_true: Sequence[int],
    candidates: Optional[Sequence[float]] = None,
) -> Tuple[float, Dict]:
    """
    Choose the high-risk threshold that maximises F1 on the given
    (train-partition) samples. Returns (threshold, train_metrics).
    """
    if candidates is None:
        candidates = [float(x) for x in range(10, 95, 5)]
    best_t, best_f1, best_cm = None, -1.0, None
    for t in candidates:
        cm = compute_confusion_matrix(y_true, [1 if s >= t else 0 for s in scores])
        f1 = cm.f1 if cm.f1 is not None else 0.0
        if f1 > best_f1:
            best_t, best_f1, best_cm = t, f1, cm
    if best_t is None or best_cm is None:
        raise ValueError("No candidate threshold produced a usable confusion matrix")
    return best_t, {"f1_on_selection_partition": round(best_f1, 4),
                    "confusion_on_selection_partition": best_cm.to_dict()}


# --------------------------------------------------------------------------
# Validation report
# --------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """
    Self-describing record of one validation run.

    ``status`` is "ok" only when real observations and real scores were
    actually compared; otherwise it is "unavailable" with the reason in
    ``message`` (e.g. FIRMS key missing) and no metrics are filled in.
    """

    status: str
    message: Optional[str] = None
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    data_period: Optional[Dict] = None            # {"start": ..., "end": ...}
    geographic_coverage: Optional[Dict] = None    # {"bbox": [w, s, e, n]}
    data_sources: Dict = field(default_factory=dict)
    model: Dict = field(default_factory=dict)     # name, version, threshold...
    temporal_separation: Optional[Dict] = None
    sample_counts: Optional[Dict] = None
    confusion_matrix: Optional[Dict] = None
    metrics: Optional[Dict] = None
    calibration: Optional[List[Dict]] = None
    error_analysis: Optional[Dict] = None
    learning: Optional[Dict] = None
    assumptions: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "status": self.status,
            "message": self.message,
            "generated_at": self.generated_at,
            "data_period": self.data_period,
            "geographic_coverage": self.geographic_coverage,
            "data_sources": self.data_sources,
            "model": self.model,
            "temporal_separation": self.temporal_separation,
            "sample_counts": self.sample_counts,
            "confusion_matrix": self.confusion_matrix,
            "metrics": self.metrics,
            "calibration": self.calibration,
            "error_analysis": self.error_analysis,
            "learning": self.learning,
            "assumptions": self.assumptions,
            "limitations": self.limitations,
        }

    def save(self, path: str) -> str:
        import os

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return path


# --------------------------------------------------------------------------
# Evaluation driver
# --------------------------------------------------------------------------

def evaluate_scores(
    scores: Sequence[float],
    y_true: Sequence[int],
    dates: Sequence[str],
    threshold: float = 65.0,
    train_fraction: float = 0.6,
    auto_threshold: bool = False,
) -> Tuple[ConfusionMatrix, Dict]:
    """
    Evaluate risk scores against observed labels with temporal separation.

    The evaluation confusion matrix is computed on the *later* evaluation
    partition only. When ``auto_threshold`` is set, the threshold is tuned
    on the earlier train partition (never on the evaluation data).

    Returns (confusion_matrix_on_evaluation, details) where details carries
    the split description, the threshold used, calibration bins and the
    Brier score on the evaluation partition.
    """
    if not (len(scores) == len(y_true) == len(dates)):
        raise ValueError("scores, y_true and dates must have equal length")
    if not scores:
        raise ValueError("No samples to evaluate")

    train_idx, test_idx = temporal_train_test_split(dates, train_fraction)

    used_threshold = float(threshold)
    threshold_info: Dict = {"value": used_threshold, "selection": "fixed"}
    if auto_threshold:
        used_threshold, sel = select_threshold(
            [scores[i] for i in train_idx], [y_true[i] for i in train_idx]
        )
        threshold_info = {"value": used_threshold, "selection": "max-F1 on train partition",
                          **sel}

    y_test = [y_true[i] for i in test_idx]
    s_test = [scores[i] for i in test_idx]
    cm = compute_confusion_matrix(y_test, [1 if s >= used_threshold else 0 for s in s_test])

    details = {
        "threshold": threshold_info,
        "train_fraction": train_fraction,
        "train_samples": len(train_idx),
        "evaluation_samples": len(test_idx),
        "train_period": [min(dates[i] for i in train_idx), max(dates[i] for i in train_idx)],
        "evaluation_period": [min(dates[i] for i in test_idx), max(dates[i] for i in test_idx)],
        "calibration": compute_calibration(s_test, y_test),
        "brier_score": brier_score(s_test, y_test),
    }
    return cm, details


# --------------------------------------------------------------------------
# Error analysis — understand WHERE and WHY the model was wrong
# --------------------------------------------------------------------------

DANGER_VS_OCCURRENCE_NOTE = (
    "Talaix predicts fire DANGER, not fire OCCURRENCE: high danger "
    "means conditions favour spread IF an ignition happens. A false "
    "positive is therefore not automatically a model failure — it can be a "
    "correct danger assessment on a day without ignition."
)


def explain_error(error_type: str, features: Dict, score: float,
                  threshold: float) -> str:
    """
    Generate a per-sample explanation of a classification outcome from the
    sample's real features. Every cited value is the actual measured /
    reanalysis value of that sample.
    """
    fwi = features.get("fwi")
    wind = features.get("wind_max_kmh")
    rain = features.get("precip_mm")
    rh = features.get("rh_mean_pct")
    fwi_s = f"{fwi:.1f}" if isinstance(fwi, (int, float)) else "n/a"

    if error_type == "fp":
        reasons = [f"Fire-weather danger was high (FWI {fwi_s})"]
        if isinstance(rh, (int, float)) and rh < 35:
            reasons.append(f"humidity was very low ({rh:.0f}%)")
        if isinstance(wind, (int, float)) and wind >= 25:
            reasons.append(f"winds were strong ({wind:.0f} km/h)")
        reasons.append("but no ignition was detected — high fire danger does "
                       "not guarantee ignition (danger ≠ occurrence)")
        return "; ".join(reasons) + "."
    if error_type == "fn":
        reasons = [f"A fire occurred although the score was below the "
                   f"threshold ({score:.1f} < {threshold:.0f}; FWI {fwi_s})"]
        if isinstance(rain, (int, float)) and rain > 2:
            reasons.append(f"it happened despite {rain:.1f} mm rain that day "
                           "— possible drought-legacy fuels or a detection "
                           "from before the rain")
        if isinstance(wind, (int, float)) and wind >= 25:
            reasons.append(f"spread was favoured by strong wind ({wind:.0f} "
                           "km/h), which the FWI-anchored score underweights")
        if len(reasons) == 1:
            reasons.append("no obvious mitigating factor in the daily "
                           "aggregates — candidate for feature review")
        return "; ".join(reasons) + "."
    if error_type == "tp":
        return (f"Correctly flagged: fire detected on a high-danger day "
                f"(FWI {fwi_s}, score {score:.1f}).")
    return (f"Correctly dismissed: no fire detected and danger was low "
            f"(FWI {fwi_s}, score {score:.1f}).")


def _classification_confidence(score: float, threshold: float) -> str:
    """How decisive the classification was (declared margin heuristic)."""
    margin = abs(score - threshold)
    return "high" if margin >= 20.0 else "moderate" if margin >= 10.0 else "low"


def _lesson_for(error_type: str) -> str:
    """The recorded lesson per outcome type (feeds 'where does Talaix fail?')."""
    return {
        "fp": "High-danger day without ignition: the danger assessment may be "
              "correct while the occurrence label is negative — pair danger with "
              "an ignition-likelihood layer before any occurrence claim.",
        "fn": "Fire under sub-threshold danger: investigate local fuel continuity, "
              "ignition sources and wind underweighting in the screening score.",
        "tp": "Danger and occurrence aligned — supports the FWI anchoring.",
        "tn": "Low danger and no occurrence aligned.",
    }[error_type]


def analyze_errors(
    scores: Sequence[float],
    y_true: Sequence[int],
    dates: Sequence[str],
    features_list: Sequence[Dict],
    threshold: float = 65.0,
    lats: Optional[Sequence[float]] = None,
    lons: Optional[Sequence[float]] = None,
    model_version: str = "1.0.0",
) -> Dict:
    """
    Per-sample error records plus pattern aggregation.

    Patterns are computed over:
      - season (calendar month),
      - fire-weather conditions (FWI bins),
      - geography (quadrants split at the median lat/lon of the samples).

    Returns per-sample records (type + generated explanation citing the real
    feature values) and per-pattern counts. Only patterns with >= 3 samples
    are reported (smaller groups carry no evidence).
    """
    if not (len(scores) == len(y_true) == len(dates) == len(features_list)):
        raise ValueError("scores, y_true, dates and features_list must have equal length")

    records = []
    for i in range(len(scores)):
        pred = 1 if scores[i] >= threshold else 0
        t = int(bool(y_true[i]))
        if pred and t:
            etype = "tp"
        elif pred and not t:
            etype = "fp"
        elif not pred and not t:
            etype = "tn"
        else:
            etype = "fn"
        rec = {
            "index": i,
            "date": dates[i],
            "score": scores[i],
            "label": t,
            "type": etype,
            "explanation": explain_error(etype, features_list[i], scores[i], threshold),
            "confidence": _classification_confidence(scores[i], threshold),
            "lesson": _lesson_for(etype),
            "model_version": model_version,
            "features": {k: v for k, v in features_list[i].items()},
        }
        if lats is not None and lons is not None:
            rec["lat"], rec["lon"] = lats[i], lons[i]
        records.append(rec)

    def _pattern(key_fn):
        groups: Dict[str, Dict[str, int]] = {}
        for i, rec in enumerate(records):
            g = key_fn(i)
            if g is None:
                continue
            bucket = groups.setdefault(g, {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "total": 0})
            bucket[rec["type"]] += 1
            bucket["total"] += 1
        return {g: c for g, c in sorted(groups.items()) if c["total"] >= 3}

    by_month = _pattern(lambda i: dates[i][:7])
    by_fwi_bin = _pattern(
        lambda i: (
            "FWI<11" if (features_list[i].get("fwi") or 0) < 11.2 else
            "FWI 11-21" if (features_list[i].get("fwi") or 0) < 21.3 else
            "FWI 21-38" if (features_list[i].get("fwi") or 0) < 38.0 else
            "FWI 38+"
        )
    )
    by_geography = None
    if lats is not None and lons is not None:
        med_lat = sorted(lats)[len(lats) // 2]
        med_lon = sorted(lons)[len(lons) // 2]
        by_geography = _pattern(
            lambda i: (
                ("N" if lats[i] >= med_lat else "S") + ("E" if lons[i] >= med_lon else "W")
            )
        )

    errors = [r for r in records if r["type"] in ("fp", "fn")]
    return {
        "records": records,
        "n_errors": len(errors),
        "false_positives": [r for r in errors if r["type"] == "fp"],
        "false_negatives": [r for r in errors if r["type"] == "fn"],
        "patterns": {
            "by_month": by_month,
            "by_fwi_bin": by_fwi_bin,
            "by_geography": by_geography,
        },
        "danger_vs_occurrence_note": DANGER_VS_OCCURRENCE_NOTE,
    }


# --------------------------------------------------------------------------
# Calibration learning — learn the score->frequency mapping from errors
# --------------------------------------------------------------------------

def fit_score_calibration(
    scores: Sequence[float],
    y_true: Sequence[int],
    n_bins: int = 10,
    score_max: float = 100.0,
    smoothing: float = 1.0,
) -> Dict:
    """
    Learn an empirical score-bin -> observed-fire-frequency mapping.

    Fit ONLY on a train partition (never on evaluation data). Uses Laplace
    smoothing so empty bins fall back gracefully. The mapping converts the
    0-100 screening score into an empirical fire-occurrence frequency — a
    learned calibration, recorded with its training provenance.
    """
    if len(scores) != len(y_true):
        raise ValueError("scores and y_true must have equal length")
    if not scores:
        raise ValueError("No samples to fit calibration")
    bins = compute_calibration(scores, y_true, n_bins=n_bins, score_max=score_max)
    width = score_max / n_bins
    mapping = []
    for b in range(n_bins):
        entry = next((x for x in bins if int(x["lower"] / width) == b), None)
        if entry:
            n = entry["count"]
            freq = (entry["observed_frequency"] * n + smoothing * 0.5) / (n + smoothing)
            mapping.append({"lower": entry["lower"], "upper": entry["upper"],
                            "count": n, "frequency": round(freq, 4)})
        else:
            mapping.append({"lower": round(b * width, 4), "upper": round((b + 1) * width, 4),
                            "count": 0, "frequency": None})
    global_freq = sum(int(bool(t)) for t in y_true) / len(y_true)
    return {
        "type": "empirical_bin_calibration",
        "n_bins": n_bins,
        "score_max": score_max,
        "bins": mapping,
        "global_frequency": round(global_freq, 4),
        "n_train": len(scores),
        "smoothing": smoothing,
    }


def apply_calibration(scores: Sequence[float], mapping: Dict) -> List[float]:
    """Map 0-100 scores to calibrated frequencies; empty bins use the global rate."""
    width = mapping["score_max"] / mapping["n_bins"]
    out = []
    for s in scores:
        b = max(0, min(mapping["n_bins"] - 1, int(float(s) / width)))
        entry = mapping["bins"][b]
        out.append(entry["frequency"] if entry["frequency"] is not None
                   else mapping["global_frequency"])
    return out


def calibration_improvement(
    train_scores: Sequence[float], train_labels: Sequence[int],
    eval_scores: Sequence[float], eval_labels: Sequence[int],
) -> Dict:
    """
    Learn calibration on the train partition, measure Brier-score change on
    the (held-out) evaluation partition. Honest learning evidence: both the
    before and after numbers are reported on data the fit never saw.
    """
    mapping = fit_score_calibration(train_scores, train_labels)
    calibrated = apply_calibration(eval_scores, mapping)
    before = brier_score(eval_scores, eval_labels)
    after = brier_score([c * 100.0 for c in calibrated], eval_labels)
    return {
        "mapping": mapping,
        "brier_before": before,
        "brier_after": after,
        "improved": (after is not None and before is not None and after < before),
        "note": "Calibration learned on the train partition only; Brier "
                "scores computed on the held-out evaluation partition.",
    }


# --------------------------------------------------------------------------
# Model registry — versioned, never auto-promoted
# --------------------------------------------------------------------------

class ModelRegistry:
    """
    JSON-file registry of model versions and their validation evidence.

    Promoting a version to production is a deliberate, manual decision —
    this registry only records evidence; nothing here changes serving code.
    """

    def __init__(self, path: str) -> None:
        self.path = path

    def _load(self) -> Dict:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return {"versions": []}

    def _save(self, data: Dict) -> None:
        import os

        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    def record(self, entry: Dict) -> Dict:
        """Record a model/validation entry (status is 'candidate' unless set)."""
        data = self._load()
        entry = dict(entry)
        entry.setdefault("status", "candidate")
        entry.setdefault("registered_at", datetime.utcnow().isoformat() + "Z")
        data["versions"].append(entry)
        self._save(data)
        return entry

    def list(self) -> List[Dict]:
        return self._load().get("versions", [])
