"""Predictor: probabilistic failure forecasting for the repair agent.

Architectural Rationale
-----------------------
The previous Predictor was a pattern detector: it looked for specific
signals (increasing gaps, high frequency, repeated outages) and returned
a binary alert flag.  This is useful but reactive — it detects patterns
that have already occurred.

The upgraded Predictor adds probabilistic forecasting (Requirement 8):

1. **Likely failures**: What anomaly type is most likely to occur next?
2. **Expected outage time**: How long will the next outage likely last?
3. **Replica failure probability**: Based on node health trend analysis.
4. **SLA breach probability**: Given current trends, will SLA be violated?
5. **Cascading failure probability**: Could one failure trigger others?

Design decisions
----------------
- All five forecasts are computed independently and combined into a
  ``ProbabilisticForecast`` dataclass alongside the original pattern
  detection results.
- Forecasting is implemented using lightweight statistical methods
  (linear regression, exponential smoothing) so no ML dependencies
  are required.
- The ``MLForecastAdapter`` interface is defined as a stub so future
  integrations (scikit-learn, Prophet, PyTorch) can be plugged in
  by implementing one method.
- The Predictor is read-only: it never modifies any table.
- All original methods (analyse, PredictionReport) are preserved for
  backward compatibility.
"""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from db.client import get_db_connection

from logging_config import logger


# ---------------------------------------------------------------------------
# PredictionReport (original — preserved for backward compat)
# ---------------------------------------------------------------------------

@dataclass
class PredictionReport:
    """Summary of detected failure patterns and predicted risk.

    Attributes
    ----------
    run_id              : Current run identifier.
    predicted_risk      : Aggregate predicted risk [0.0, 1.0].
    patterns_detected   : List of (pattern_name, signal_strength, description).
    recommended_action  : Suggested proactive action (advisory only).
    created_at          : UTC timestamp.
    """
    run_id: str
    predicted_risk: float
    patterns_detected: List[Dict[str, Any]] = field(default_factory=list)
    recommended_action: str = "none"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def is_alert(self) -> bool:
        """True if the predicted risk exceeds the alert threshold (0.6)."""
        return self.predicted_risk >= 0.6


# ---------------------------------------------------------------------------
# ProbabilisticForecast (new — Requirement 8)
# ---------------------------------------------------------------------------

@dataclass
class ProbabilisticForecast:
    """Probabilistic failure forecast from the Predictor.

    Contains five forward-looking probability estimates alongside the
    original pattern-detection results.

    Attributes
    ----------
    run_id                     : Current pipeline run ID.
    likely_failure_type        : Most probable next anomaly type.
    likely_failure_probability : P(failure in next window) [0,1].
    expected_outage_minutes    : Expected duration if failure occurs.
    replica_failure_prob       : P(replica node failure) [0,1].
    sla_breach_prob            : P(SLA breach in next period) [0,1].
    cascading_failure_prob     : P(cascading failure from current state) [0,1].
    forecast_horizon_minutes   : Look-ahead window for these forecasts.
    confidence                 : Forecast confidence [0,1].
    contributing_patterns      : Patterns that influenced this forecast.
    recommendation             : Proactive action recommendation.
    created_at                 : UTC timestamp.
    """
    run_id: str
    likely_failure_type: str = "none"
    likely_failure_probability: float = 0.0
    expected_outage_minutes: float = 0.0
    replica_failure_prob: float = 0.0
    sla_breach_prob: float = 0.0
    cascading_failure_prob: float = 0.0
    forecast_horizon_minutes: int = 60
    confidence: float = 0.5
    contributing_patterns: List[str] = field(default_factory=list)
    recommendation: str = "none"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def is_high_risk(self) -> bool:
        """True if any single probability exceeds 0.65."""
        return any([
            self.likely_failure_probability > 0.65,
            self.replica_failure_prob > 0.65,
            self.sla_breach_prob > 0.65,
            self.cascading_failure_prob > 0.65,
        ])

    @property
    def overall_risk(self) -> float:
        """Weighted aggregate of all five probability estimates."""
        return round(
            0.30 * self.likely_failure_probability
            + 0.25 * self.replica_failure_prob
            + 0.25 * self.sla_breach_prob
            + 0.20 * self.cascading_failure_prob,
            4,
        )


# ---------------------------------------------------------------------------
# ML Forecast Adapter (extensibility interface — Requirement 16)
# ---------------------------------------------------------------------------

class MLForecastAdapter(ABC):
    """Abstract interface for plugging in an external ML forecaster.

    Implement this class to integrate any time-series forecasting model:
    - ``ProphetAdapter``: Facebook Prophet for seasonality-aware forecasting.
    - ``SKLearnAdapter``: scikit-learn regression/classification.
    - ``PyTorchAdapter``: LSTM or Transformer-based sequence model.
    - ``AutoMLAdapter``: AutoGluon or TPOT for automated model selection.

    Usage::

        adapter = ProphetAdapter(model_path='models/failure_prophet.pkl')
        predictor = Predictor(db_path=db_path, ml_adapter=adapter)
        forecast = predictor.forecast(run_id, anomaly_type)
    """

    @abstractmethod
    def predict_failure_probability(
        self,
        recent_incidents: List[Dict[str, Any]],
        horizon_minutes: int,
    ) -> Tuple[str, float]:
        """Predict the most likely failure type and its probability.

        Args:
            recent_incidents : Recent incident data from the DB.
            horizon_minutes  : Prediction horizon in minutes.

        Returns:
            Tuple of (failure_type: str, probability: float).
        """

    @abstractmethod
    def predict_outage_duration(
        self,
        recent_incidents: List[Dict[str, Any]],
    ) -> float:
        """Predict expected outage duration in minutes.

        Args:
            recent_incidents : Recent incident data.

        Returns:
            Expected outage duration in minutes.
        """


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class Predictor:
    """Probabilistic failure forecaster for the autonomous repair agent.

    Combines the original pattern-detection approach with five new
    probabilistic forward-looking forecasts.

    Parameters
    ----------
    db_path    : str
        Path to the SQLite database.
    ml_adapter : MLForecastAdapter, optional
        External ML model adapter for enhanced forecasting.

    Usage::

        predictor = Predictor(db_path=db_path)

        # Original API (backward compat)
        report = predictor.analyse(run_id, anomaly_type)

        # New probabilistic forecast
        forecast = predictor.forecast(run_id, anomaly_type)
    """

    LOOKBACK_DAYS: int = 7
    MIN_INCIDENTS_FOR_PATTERN: int = 3
    FREQUENCY_INSTABILITY_THRESHOLD: float = 0.5
    NULL_RATE_TREND_THRESHOLD: float = 0.02
    GAP_TREND_SLOPE_THRESHOLD: float = 10.0

    def __init__(
        self,
        db_path: str,
        ml_adapter: Optional[MLForecastAdapter] = None,
    ) -> None:
        self.db_path = db_path
        self._ml_adapter = ml_adapter

    # ------------------------------------------------------------------
    # Original API (backward-compatible)
    # ------------------------------------------------------------------

    def analyse(self, run_id: str, anomaly_type: str = "") -> PredictionReport:
        """Analyse historical data and return a PredictionReport.

        Args:
            run_id:       Current pipeline run ID.
            anomaly_type: Focus analysis on this anomaly type (optional).

        Returns:
            PredictionReport with detected patterns and predicted risk.
        """
        incidents = self._load_recent_incidents(anomaly_type)
        patterns: List[Dict[str, Any]] = []

        if len(incidents) >= self.MIN_INCIDENTS_FOR_PATTERN:
            for detector in [
                self._detect_gap_trend,
                self._detect_high_frequency,
                lambda i: self._detect_repeated_outages(i, anomaly_type),
                self._detect_null_rate_trend,
            ]:
                result = detector(incidents)
                if result:
                    patterns.append(result)

        node_pattern = self._detect_node_exhaustion()
        if node_pattern:
            patterns.append(node_pattern)

        if patterns:
            risk = sum(p["signal_strength"] for p in patterns) / len(patterns)
            risk = min(risk * (1.0 + 0.1 * len(patterns)), 1.0)
        else:
            risk = 0.0

        recommended_action = self._recommend_action(patterns, risk)

        report = PredictionReport(
            run_id=run_id,
            predicted_risk=round(risk, 4),
            patterns_detected=patterns,
            recommended_action=recommended_action,
        )

        if report.is_alert:
            logger.warning(
                f"[{run_id}] Predictor: HIGH predicted_risk={risk:.3f} | "
                f"patterns={[p['name'] for p in patterns]} | "
                f"recommended={recommended_action}"
            )
        else:
            logger.info(
                f"[{run_id}] Predictor: predicted_risk={risk:.3f} | "
                f"patterns={len(patterns)} detected"
            )

        return report

    # ------------------------------------------------------------------
    # New probabilistic forecasting API (Requirement 8)
    # ------------------------------------------------------------------

    def forecast(
        self,
        run_id: str,
        anomaly_type: str = "",
        horizon_minutes: int = 60,
    ) -> ProbabilisticForecast:
        """Generate a probabilistic failure forecast.

        Computes five probability estimates:
        - Likely failure type + probability.
        - Expected outage duration.
        - Replica failure probability.
        - SLA breach probability.
        - Cascading failure probability.

        If an ML adapter is configured, uses it for failure type and
        outage duration.  Otherwise uses statistical estimation.

        Args:
            run_id          : Current pipeline run ID.
            anomaly_type    : Focus on this anomaly type (optional).
            horizon_minutes : Look-ahead window for forecasts.

        Returns:
            ProbabilisticForecast with all five probability estimates.
        """
        incidents = self._load_recent_incidents(anomaly_type)
        repair_history = self._load_repair_history(anomaly_type)

        # ── 1. Likely failure type ─────────────────────────────────────
        if self._ml_adapter and incidents:
            failure_type, failure_prob = self._ml_adapter.predict_failure_probability(
                incidents, horizon_minutes
            )
        else:
            failure_type, failure_prob = self._estimate_likely_failure(
                incidents, anomaly_type, horizon_minutes
            )

        # ── 2. Expected outage duration ────────────────────────────────
        if self._ml_adapter and incidents:
            expected_outage = self._ml_adapter.predict_outage_duration(incidents)
        else:
            expected_outage = self._estimate_outage_duration(incidents)

        # ── 3. Replica failure probability ────────────────────────────
        replica_prob = self._estimate_replica_failure_prob(incidents)

        # ── 4. SLA breach probability ─────────────────────────────────
        sla_breach_prob = self._estimate_sla_breach_prob(
            incidents, repair_history, expected_outage
        )

        # ── 5. Cascading failure probability ──────────────────────────
        cascading_prob = self._estimate_cascading_failure_prob(
            incidents, replica_prob
        )

        # ── Contributing patterns (for explainability) ─────────────────
        patterns = self._detect_all_patterns(incidents, anomaly_type)
        contributing = [p["name"] for p in patterns]

        # ── Forecast confidence ────────────────────────────────────────
        confidence = self._estimate_forecast_confidence(
            len(incidents), len(repair_history)
        )

        # ── Recommendation ─────────────────────────────────────────────
        recommendation = self._forecast_recommendation(
            failure_prob, replica_prob, sla_breach_prob, cascading_prob, patterns
        )

        forecast = ProbabilisticForecast(
            run_id=run_id,
            likely_failure_type=failure_type,
            likely_failure_probability=round(failure_prob, 4),
            expected_outage_minutes=round(expected_outage, 2),
            replica_failure_prob=round(replica_prob, 4),
            sla_breach_prob=round(sla_breach_prob, 4),
            cascading_failure_prob=round(cascading_prob, 4),
            forecast_horizon_minutes=horizon_minutes,
            confidence=round(confidence, 4),
            contributing_patterns=contributing,
            recommendation=recommendation,
        )

        logger.info(
            f"[{run_id}] Predictor forecast: "
            f"failure_type={failure_type} "
            f"({failure_prob:.2%}) | "
            f"outage={expected_outage:.1f}min | "
            f"replica={replica_prob:.2%} | "
            f"sla_breach={sla_breach_prob:.2%} | "
            f"cascade={cascading_prob:.2%} | "
            f"overall_risk={forecast.overall_risk:.3f} | "
            f"confidence={confidence:.2f}"
        )

        return forecast

    # ------------------------------------------------------------------
    # Probabilistic estimators
    # ------------------------------------------------------------------

    def _estimate_likely_failure(
        self,
        incidents: list,
        anomaly_type: str,
        horizon_minutes: int,
    ) -> Tuple[str, float]:
        """Estimate the most likely failure type and its probability.

        Method: frequency × recency weighting over recent incidents.
        More recent incidents get higher weight (exponential decay).
        """
        if not incidents:
            return ("none", 0.05)

        type_weights: Dict[str, float] = {}
        for i, incident in enumerate(incidents):
            # Exponential decay: most recent = weight 1.0, oldest → 0
            recency_weight = 0.9 ** i
            atype = incident.get("anomaly_type", "unknown")
            type_weights[atype] = type_weights.get(atype, 0.0) + recency_weight

        # Normalise
        total = sum(type_weights.values())
        if total == 0:
            return ("none", 0.0)

        best_type = max(type_weights, key=type_weights.get)
        raw_prob = type_weights[best_type] / total

        # Scale by frequency: high frequency → higher probability in horizon
        lookback_hours = self.LOOKBACK_DAYS * 24
        frequency = len(incidents) / lookback_hours
        horizon_hours = horizon_minutes / 60.0
        adjusted_prob = min(1.0 - (1.0 - raw_prob) ** (frequency * horizon_hours), 0.95)

        return (best_type, round(adjusted_prob, 4))

    def _estimate_outage_duration(self, incidents: list) -> float:
        """Estimate expected outage duration using exponential smoothing."""
        gaps = [i.get("gap_minutes", 0) for i in incidents if i.get("gap_minutes", 0) > 0]
        if not gaps:
            return 15.0  # default: 15 min

        # Exponential smoothing (alpha=0.3 — more weight on recent)
        alpha = 0.3
        smoothed = gaps[0]
        for g in gaps[1:]:
            smoothed = alpha * g + (1.0 - alpha) * smoothed

        return round(smoothed, 2)

    def _estimate_replica_failure_prob(self, incidents: list) -> float:
        """Estimate probability of replica node failure.

        Factors: node exhaustion pattern + incident gap trend.
        """
        # Check node health
        node_exhaustion_signal = 0.0
        try:
            from db.database_manager import get_all_nodes_status
            statuses = get_all_nodes_status()
            failed = sum(1 for n in statuses if n["status"] == "failed")
            total = len(statuses)
            if total > 0:
                node_exhaustion_signal = failed / total
        except Exception:
            node_exhaustion_signal = 0.2  # neutral if unavailable

        # Gap trend signal
        gaps = [i.get("gap_minutes", 0) for i in incidents]
        gap_trend_signal = 0.0
        if len(gaps) >= 3:
            slope = self._linear_slope(gaps)
            gap_trend_signal = min(slope / 60.0, 0.5) if slope > 0 else 0.0

        # Combined estimate
        replica_prob = 0.5 * node_exhaustion_signal + 0.5 * gap_trend_signal
        return round(min(replica_prob, 0.95), 4)

    def _estimate_sla_breach_prob(
        self,
        incidents: list,
        repair_history: list,
        expected_outage_minutes: float,
    ) -> float:
        """Estimate probability of SLA breach in the next period.

        Factors:
        - Expected outage duration vs. SLA threshold (assume 30 min SLA).
        - Historical repair failure rate.
        - Current incident frequency.
        """
        SLA_THRESHOLD_MINUTES = 30.0

        # Duration factor: if expected outage > SLA → near-certain breach
        duration_factor = min(expected_outage_minutes / SLA_THRESHOLD_MINUTES, 1.0)

        # Repair failure factor
        if repair_history:
            failures = sum(1 for r in repair_history if r.get("final_outcome") == "failure")
            failure_rate = failures / len(repair_history)
        else:
            failure_rate = 0.3  # neutral prior

        # Frequency factor: high frequency → higher SLA risk
        lookback_hours = self.LOOKBACK_DAYS * 24
        frequency = len(incidents) / max(lookback_hours, 1)
        frequency_factor = min(frequency / 1.0, 1.0)  # 1 per hour = max

        sla_prob = (
            0.45 * duration_factor
            + 0.35 * failure_rate
            + 0.20 * frequency_factor
        )
        return round(min(sla_prob, 0.95), 4)

    def _estimate_cascading_failure_prob(
        self,
        incidents: list,
        replica_prob: float,
    ) -> float:
        """Estimate probability of cascading failure.

        A cascading failure is likely when:
        - Replica failure probability is high (no fallback).
        - Multiple anomaly types appear in recent incidents.
        - Unresolved incidents are accumulating.
        """
        # Type diversity: multiple anomaly types → higher cascade risk
        anomaly_types = set(i.get("anomaly_type", "") for i in incidents)
        diversity_factor = min(len(anomaly_types) / 3.0, 1.0)

        # Unresolved fraction
        if incidents:
            unresolved = sum(1 for i in incidents if not i.get("resolved"))
            unresolved_factor = unresolved / len(incidents)
        else:
            unresolved_factor = 0.0

        cascade_prob = (
            0.40 * replica_prob
            + 0.35 * unresolved_factor
            + 0.25 * diversity_factor
        )
        return round(min(cascade_prob, 0.90), 4)

    def _estimate_forecast_confidence(
        self,
        incident_count: int,
        repair_count: int,
    ) -> float:
        """Estimate confidence in the forecast based on data availability.

        More historical data → higher confidence.
        """
        # Logarithmic confidence: plateaus at ~20 data points
        import math
        data_points = incident_count + repair_count
        if data_points == 0:
            return 0.20
        confidence = min(0.20 + 0.60 * math.log(data_points + 1) / math.log(25), 0.90)
        return round(confidence, 4)

    # ------------------------------------------------------------------
    # Original pattern detectors (preserved unchanged)
    # ------------------------------------------------------------------

    def _detect_gap_trend(self, incidents: list) -> Optional[Dict[str, Any]]:
        """Detect an increasing gap_minutes trend."""
        gaps = [i.get("gap_minutes", 0) for i in incidents if i.get("gap_minutes")]
        if len(gaps) < 3:
            return None
        slope = self._linear_slope(gaps)
        if slope < self.GAP_TREND_SLOPE_THRESHOLD:
            return None
        signal = min(slope / 60.0, 1.0)
        return {
            "name": "increasing_gap_trend",
            "signal_strength": round(signal, 3),
            "description": (
                f"Gap duration increasing at {slope:.1f} min/incident over "
                f"last {len(incidents)} incidents — replica lag probable"
            ),
        }

    def _detect_high_frequency(self, incidents: list) -> Optional[Dict[str, Any]]:
        """Detect DB instability via high incident frequency."""
        if not incidents:
            return None
        lookback_hours = self.LOOKBACK_DAYS * 24
        frequency = len(incidents) / lookback_hours
        if frequency < self.FREQUENCY_INSTABILITY_THRESHOLD:
            return None
        signal = min(frequency / 2.0, 1.0)
        return {
            "name": "db_instability",
            "signal_strength": round(signal, 3),
            "description": (
                f"{len(incidents)} incidents in {self.LOOKBACK_DAYS}d "
                f"({frequency:.2f}/h) — database instability detected"
            ),
        }

    def _detect_repeated_outages(
        self, incidents: list, anomaly_type: str
    ) -> Optional[Dict[str, Any]]:
        """Detect repeated outages of the same type."""
        same_type = [i for i in incidents if i.get("anomaly_type") == anomaly_type]
        if len(same_type) < self.MIN_INCIDENTS_FOR_PATTERN:
            return None
        unresolved = sum(1 for i in same_type if not i.get("resolved"))
        unresolved_rate = unresolved / len(same_type)
        signal = 0.4 + unresolved_rate * 0.6
        return {
            "name": "repeated_outages",
            "signal_strength": round(signal, 3),
            "description": (
                f"{len(same_type)} {anomaly_type} incidents in last "
                f"{self.LOOKBACK_DAYS}d; {unresolved} unresolved — "
                "recurring failure pattern"
            ),
        }

    def _detect_null_rate_trend(self, incidents: list) -> Optional[Dict[str, Any]]:
        """Detect an upward trend in null rates."""
        dq_incidents = [i for i in incidents if i.get("anomaly_type") == "data_quality"]
        if len(dq_incidents) < 2:
            return None
        confidences = [i.get("confidence", 0.5) for i in dq_incidents]
        if len(confidences) >= 2:
            trend = confidences[0] - confidences[-1]
            if trend < -0.1:
                signal = min(abs(trend), 1.0)
                return {
                    "name": "growing_null_rate",
                    "signal_strength": round(signal, 3),
                    "description": (
                        f"Data quality confidence degrading "
                        f"({confidences[-1]:.2f} → {confidences[0]:.2f}) — "
                        "null rate may be increasing"
                    ),
                }
        return None

    def _detect_node_exhaustion(self) -> Optional[Dict[str, Any]]:
        """Detect that available DB replicas are running low."""
        try:
            from db.database_manager import get_all_nodes_status
            statuses = get_all_nodes_status()
            failed = sum(1 for n in statuses if n["status"] == "failed")
            total  = len(statuses)
            if total == 0 or failed == 0:
                return None
            failed_fraction = failed / total
            if failed_fraction < 0.5:
                return None
            signal = min(failed_fraction, 1.0)
            return {
                "name": "node_exhaustion",
                "signal_strength": round(signal, 3),
                "description": (
                    f"{failed}/{total} DB nodes failed — "
                    "replica exhaustion: manual recovery needed"
                ),
            }
        except Exception as exc:
            logger.debug(f"Predictor._detect_node_exhaustion: {exc}")
            return None

    def _detect_all_patterns(
        self, incidents: list, anomaly_type: str
    ) -> List[Dict[str, Any]]:
        """Run all pattern detectors and return all detected patterns."""
        patterns = []
        if len(incidents) >= self.MIN_INCIDENTS_FOR_PATTERN:
            for detector in [
                self._detect_gap_trend,
                self._detect_high_frequency,
                lambda i: self._detect_repeated_outages(i, anomaly_type),
                self._detect_null_rate_trend,
            ]:
                r = detector(incidents)
                if r:
                    patterns.append(r)
        node = self._detect_node_exhaustion()
        if node:
            patterns.append(node)
        return patterns

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------

    def _recommend_action(
        self, patterns: List[Dict[str, Any]], risk: float
    ) -> str:
        """Suggest a proactive action based on detected patterns and risk."""
        names = {p["name"] for p in patterns}
        if "node_exhaustion" in names:
            return "reset_cluster"
        if "repeated_outages" in names and risk > 0.7:
            return "escalate_to_human"
        if "increasing_gap_trend" in names:
            return "pre_populate_backup"
        if "db_instability" in names:
            return "increase_monitoring_frequency"
        if "growing_null_rate" in names:
            return "enable_strict_validation"
        if risk > 0.5:
            return "increase_monitoring_frequency"
        return "none"

    def _forecast_recommendation(
        self,
        failure_prob: float,
        replica_prob: float,
        sla_breach_prob: float,
        cascading_prob: float,
        patterns: List[Dict[str, Any]],
    ) -> str:
        """Generate a proactive recommendation from forecast probabilities."""
        if cascading_prob > 0.70:
            return "escalate_to_human"
        if replica_prob > 0.65:
            return "pre_populate_backup"
        if sla_breach_prob > 0.70:
            return "increase_monitoring_frequency"
        if failure_prob > 0.65:
            return "pre_warm_backup_node"
        return self._recommend_action(patterns, failure_prob)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _linear_slope(values: List[float]) -> float:
        """Compute linear regression slope for a sequence of values."""
        n = len(values)
        if n < 2:
            return 0.0
        xs = list(range(n))
        x_mean = sum(xs) / n
        y_mean = sum(values) / n
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values))
        denominator = sum((x - x_mean) ** 2 for x in xs)
        return numerator / denominator if denominator != 0 else 0.0

    def _load_recent_incidents(self, anomaly_type: str = "") -> list:
        """Load incidents from the last LOOKBACK_DAYS days."""
        try:
            conn = get_db_connection()
            cutoff = (
                datetime.now(timezone.utc) - timedelta(days=self.LOOKBACK_DAYS)
            ).isoformat()
            if anomaly_type:
                rows = conn.execute(
                    """
                    SELECT run_id, timestamp, anomaly_type, severity,
                           gap_minutes, resolved, confidence
                    FROM incidents
                    WHERE timestamp > ? AND anomaly_type = ?
                    ORDER BY timestamp DESC
                    """,
                    (cutoff, anomaly_type),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT run_id, timestamp, anomaly_type, severity,
                           gap_minutes, resolved, confidence
                    FROM incidents
                    WHERE timestamp > ?
                    ORDER BY timestamp DESC
                    """,
                    (cutoff,),
                ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning(f"Predictor._load_recent_incidents: {exc}")
            return []

    def _load_repair_history(self, anomaly_type: str = "") -> list:
        """Load repair_memory entries for trend analysis."""
        try:
            conn = get_db_connection()
            cutoff = (
                datetime.now(timezone.utc) - timedelta(days=self.LOOKBACK_DAYS)
            ).isoformat()
            if anomaly_type:
                rows = conn.execute(
                    """
                    SELECT strategy_name, final_outcome, verification_score,
                           recovery_time_secs, created_at
                    FROM repair_memory
                    WHERE created_at > ? AND anomaly_type = ?
                    ORDER BY created_at DESC
                    """,
                    (cutoff, anomaly_type),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT strategy_name, final_outcome, verification_score,
                           recovery_time_secs, created_at
                    FROM repair_memory
                    WHERE created_at > ?
                    ORDER BY created_at DESC
                    """,
                    (cutoff,),
                ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning(f"Predictor._load_repair_history: {exc}")
            return []
