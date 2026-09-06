"""
Adaptive water intervention planning.

Determines where, when, and how much water to deploy for subsurface hydration
barriers, with a strict Human-in-the-Loop Decision Gate.

The AI generates a traffic-light recommendation for each protection zone:
    - Green (Deploy):   High confidence, low uncertainty.
    - Yellow (Review):  Moderate uncertainty, operator reviews.
    - Red (Hold):       High uncertainty or conflicting data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class InterventionPlan:
    """A single zone intervention plan."""

    zone_id: str
    water_volume_m3: float
    start_time_h: float
    duration_h: float
    recommendation: str  # 'green', 'yellow', 'red'
    confidence: float

    def to_dict(self) -> Dict[str, object]:
        """Return plan as a dictionary."""
        return {
            "zone_id": self.zone_id,
            "water_volume_m3": self.water_volume_m3,
            "start_time_h": self.start_time_h,
            "duration_h": self.duration_h,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
        }


@dataclass
class InterventionPlanner:
    """
    Plan adaptive water interventions across protection zones.

    Parameters
    ----------
    human_decision_window_h : float
        Configurable window between model recommendation and physical
        activation (hours).
    confidence_threshold_green : float
        Confidence above which a zone is 'green' (deploy).
    confidence_threshold_yellow : float
        Confidence above which a zone is 'yellow' (review).
    """

    human_decision_window_h: float = 6.0
    confidence_threshold_green: float = 0.8
    confidence_threshold_yellow: float = 0.5

    def classify_recommendation(self, confidence: float) -> str:
        """
        Classify a zone into a traffic-light recommendation.

        Parameters
        ----------
        confidence : float
            Model confidence in [0, 1].

        Returns
        -------
        str
            'green', 'yellow', or 'red'.
        """
        if confidence >= self.confidence_threshold_green:
            return "green"
        if confidence >= self.confidence_threshold_yellow:
            return "yellow"
        return "red"

    def compute_duration(
        self,
        water_volume_m3: float,
        application_rate_m3_per_h: float,
    ) -> float:
        """
        Compute intervention duration (hours).

        Parameters
        ----------
        water_volume_m3 : float
            Water volume to apply (m^3).
        application_rate_m3_per_h : float
            Application rate (m^3/hour).

        Returns
        -------
        float
            Duration in hours.
        """
        if application_rate_m3_per_h <= 0:
            return float("inf")
        return float(water_volume_m3 / application_rate_m3_per_h)

    def build_plan(
        self,
        zone_ids: List[str],
        water_volumes_m3: List[float],
        confidences: List[float],
        application_rate_m3_per_h: float = 50.0,
        lead_time_h: float = 12.0,
    ) -> List[InterventionPlan]:
        """
        Build intervention plans for a set of zones.

        Parameters
        ----------
        zone_ids : List[str]
            Zone identifiers.
        water_volumes_m3 : List[float]
            Water volume for each zone (m^3).
        confidences : List[float]
            Model confidence for each zone in [0, 1].
        application_rate_m3_per_h : float
            Water application rate (m^3/hour).
        lead_time_h : float
            Lead time before fire arrival (hours).

        Returns
        -------
        List[InterventionPlan]
            Intervention plans.
        """
        plans: List[InterventionPlan] = []
        for zone_id, volume, confidence in zip(
            zone_ids, water_volumes_m3, confidences
        ):
            rec = self.classify_recommendation(confidence)
            duration = self.compute_duration(volume, application_rate_m3_per_h)
            # Start time accounts for the human decision window
            start_time = max(0.0, lead_time_h - duration - self.human_decision_window_h)
            plans.append(
                InterventionPlan(
                    zone_id=zone_id,
                    water_volume_m3=float(volume),
                    start_time_h=float(start_time),
                    duration_h=float(duration),
                    recommendation=rec,
                    confidence=float(confidence),
                )
            )
        return plans

    def evacuation_safety_margin(
        self,
        evacuation_window_h: float,
        fire_arrival_h: float,
        operational_margin_h: float,
        uncertainty_h: float,
    ) -> float:
        """
        Compute the Evacuation Safety Margin (ESM).

            ESM = t_evacuation_window - t_fire_arrival
                  - t_operational_margin - t_uncertainty

        Parameters
        ----------
        evacuation_window_h : float
            Time available for evacuation (hours).
        fire_arrival_h : float
            Fire arrival time (hours).
        operational_margin_h : float
            Operational margin (hours).
        uncertainty_h : float
            Uncertainty buffer (hours).

        Returns
        -------
        float
            Evacuation Safety Margin (hours). Negative indicates insufficient
            safety margin.
        """
        return float(
            evacuation_window_h
            - fire_arrival_h
            - operational_margin_h
            - uncertainty_h
        )
