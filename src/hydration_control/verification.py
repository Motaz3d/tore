"""
Hindcasting validation and feedback loop with continuous improvement.

Phase 1 verification is performed through historical hindcasting using
historical burned-area observations from EFFIS. This module computes spatial
overlap and classification metrics between predicted and observed fire extents.

The system now includes continuous verification that learns from prediction accuracy
and improves model performance over time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import warnings


@dataclass
class HindcastResult:
    """Container for a single hindcast validation result."""

    event_id: str
    iou: float
    precision: float
    recall: float
    critical_success_index: float
    accuracy: float

    def to_dict(self) -> Dict[str, float]:
        """Return result as a dictionary."""
        return {
            "event_id": self.event_id,
            "iou": self.iou,
            "precision": self.precision,
            "recall": self.recall,
            "critical_success_index": self.critical_success_index,
            "accuracy": self.accuracy,
        }


class ContinuousVerificationSystem:
    """
    Continuous verification system that improves models based on live data.
    """
    def __init__(self, model_accuracy_threshold=0.8):
        self.model_accuracy_threshold = model_accuracy_threshold
        self.performance_history = {}
        self.feedback_loops = []
        self.model_improvement_suggestions = []

    def update_model_performance(self, model_name: str, predicted: float, actual: float, timestamp: datetime):
        """
        Update model performance based on the prediction-vs-actual comparison.
        """
        error = abs(predicted - actual)
        accuracy = 1 / (1 + error)  # convert error into accuracy
        
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []
        
        self.performance_history[model_name].append({
            'timestamp': timestamp,
            'accuracy': accuracy,
            'predicted': predicted,
            'actual': actual,
            'error': error
        })
        
        # Check whether retraining is needed
        recent_performance = self.get_recent_performance(model_name, days=7)
        if recent_performance:
            avg_accuracy = np.mean([p['accuracy'] for p in recent_performance])
        else:
            avg_accuracy = 0.0
        
        if avg_accuracy < self.model_accuracy_threshold:
            return {'retrain_required': True, 'current_avg_accuracy': avg_accuracy}
        else:
            return {'retrain_required': False, 'current_avg_accuracy': avg_accuracy}
    
    def get_recent_performance(self, model_name: str, days: int = 7):
        """
        Retrieve model performance over the most recent days.
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        return [p for p in self.performance_history.get(model_name, []) 
                if p['timestamp'] > cutoff_time]
    
    def add_feedback(self, model_name: str, old_prediction: float, new_prediction: float, actual: float, feedback: str):
        """
        Add feedback to the system for continuous improvement.
        """
        feedback_entry = {
            'model': model_name,
            'timestamp': datetime.now(),
            'old_prediction': old_prediction,
            'new_prediction': new_prediction,
            'actual': actual,
            'feedback': feedback,
            'error_reduction': abs(old_prediction - actual) - abs(new_prediction - actual)
        }
        self.feedback_loops.append(feedback_entry)
        
        # If the improvement is significant, add it to the improvement suggestions
        if feedback_entry['error_reduction'] > 0.1:  # notable improvement
            improvement_suggestion = {
                'model': model_name,
                'suggested_improvement': feedback,
                'expected_benefit': feedback_entry['error_reduction'],
                'timestamp': datetime.now()
            }
            self.model_improvement_suggestions.append(improvement_suggestion)
    
    def get_model_improvement_suggestions(self) -> List[Dict]:
        """
        Retrieve model improvement suggestions.
        """
        return self.model_improvement_suggestions

    def generate_adaptive_strategy(self, current_conditions: Dict) -> Dict:
        """
        Generate an adaptive strategy based on current conditions.
        """
        # Assess current conditions and provide adaptive recommendations
        risk_level = current_conditions.get('risk', 0.5)
        fuel_moisture = current_conditions.get('fuel_moisture', 15.0)
        wind_speed = current_conditions.get('wind_speed', 10.0)
        humidity = current_conditions.get('humidity', 30.0)

        # Compute spread probability from the conditions
        spread_probability = self._calculate_spread_probability(
            fuel_moisture, wind_speed, humidity
        )

        # Build the adaptive strategy
        if risk_level > 0.7:
            strategy = {
                'intervention': 'IMMEDIATE',
                'confidence': 'HIGH',
                'recommended_actions': [
                    'Activate protection zones',
                    'Deploy water resources',
                    'Issue evacuation alert'
                ],
                'resource_allocation': 'MAXIMUM'
            }
        elif risk_level > 0.5:
            strategy = {
                'intervention': 'MONITOR',
                'confidence': 'MEDIUM',
                'recommended_actions': [
                    'Continue monitoring',
                    'Prepare resources',
                    'Update risk assessment'
                ],
                'resource_allocation': 'MODERATE'
            }
        else:
            strategy = {
                'intervention': 'STANDARD',
                'confidence': 'LOW',
                'recommended_actions': [
                    'Standard monitoring',
                    'Maintain readiness'
                ],
                'resource_allocation': 'MINIMUM'
            }
        
        return strategy
    
    def _calculate_spread_probability(self, fuel_moisture: float, wind_speed: float, humidity: float) -> float:
        """
        Compute fire spread probability from environmental conditions.
        """
        # Simplified model — production use relies on more complex models
        # Wind effect: 0.5 * (wind_speed / 10) ^ 1.5
        wind_factor = 0.5 * (wind_speed / 10) ** 1.5

        # Fuel moisture effect: exp(-0.1 * (20 - fuel_moisture))
        moisture_factor = np.exp(-0.1 * max(0, 20 - fuel_moisture))

        # Atmospheric humidity effect: exp(-0.05 * humidity)
        humidity_factor = np.exp(-0.05 * max(0, humidity))

        # Combine the factors
        combined_risk = 0.5 + 0.3 * wind_factor + 0.2 * moisture_factor
        return min(1.0, max(0.0, combined_risk))


@dataclass
class HindcastValidator:
    """
    Validate model predictions against historical burned-area observations.

    Parameters
    ----------
    threshold : float
        Probability threshold for classifying a pixel as burned.
    """

    threshold: float = 0.5

    def validate_event(
        self,
        event_id: str,
        predicted_probability: np.ndarray,
        observed_burned: np.ndarray,
    ) -> HindcastResult:
        """
        Validate a single historical fire event.

        Parameters
        ----------
        event_id : str
            Identifier of the historical fire event.
        predicted_probability : np.ndarray
            Predicted burn probability in [0, 1].
        observed_burned : np.ndarray
            Binary observed burned mask (1 = burned, 0 = not burned).

        Returns
        -------
        HindcastResult
            Validation metrics.
        """
        pred = np.asarray(predicted_probability, dtype=float)
        obs = np.asarray(observed_burned, dtype=bool)

        pred_binary = pred >= self.threshold

        tp = np.sum(pred_binary & obs)
        fp = np.sum(pred_binary & ~obs)
        fn = np.sum(~pred_binary & obs)
        tn = np.sum(~pred_binary & ~obs)

        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        return HindcastResult(
            event_id=event_id,
            iou=float(iou),
            precision=float(precision),
            recall=float(recall),
            critical_success_index=float(csi),
            accuracy=float(accuracy),
        )

    def validate_events(
        self,
        events: List[Dict[str, object]],
    ) -> List[HindcastResult]:
        """
        Validate a list of historical fire events.

        Parameters
        ----------
        events : List[Dict[str, object]]
            Each dict must contain 'id', 'predicted_probability', and
            'observed_burned'.

        Returns
        -------
        List[HindcastResult]
            Validation results for each event.
        """
        results: List[HindcastResult] = []
        for event in events:
            event_id = str(event.get("id", "unknown"))
            pred = np.asarray(event["predicted_probability"], dtype=float)
            obs = np.asarray(event["observed_burned"], dtype=bool)
            results.append(self.validate_event(event_id, pred, obs))
        return results

    def aggregate_metrics(
        self,
        results: List[HindcastResult],
    ) -> Dict[str, float]:
        """
        Aggregate metrics across multiple events.

        Parameters
        ----------
        results : List[HindcastResult]
            Individual event validation results.

        Returns
        -------
        Dict[str, float]
            Mean metrics across events.
        """
        if not results:
            return {
                "mean_iou": 0.0,
                "mean_precision": 0.0,
                "mean_recall": 0.0,
                "mean_csi": 0.0,
                "mean_accuracy": 0.0,
            }
        return {
            "mean_iou": float(np.mean([r.iou for r in results])),
            "mean_precision": float(np.mean([r.precision for r in results])),
            "mean_recall": float(np.mean([r.recall for r in results])),
            "mean_csi": float(np.mean([r.critical_success_index for r in results])),
            "mean_accuracy": float(np.mean([r.accuracy for r in results])),
        }