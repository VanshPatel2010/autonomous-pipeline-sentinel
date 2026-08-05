"""Monitor Agent: polls SQLite for pipeline health every 5 minutes.

Checks:
- Row count vs 7-day rolling baseline
- Null rate on critical columns (order_amount)
- Data staleness (freshness window)
- Gap duration estimation

Writes findings into the LangGraph state dict (STM).
No LLM calls — purely statistical anomaly detection.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from db.client import get_db_connection, DBWrapper

import pandas as pd

from config import (
    ANOMALY_THRESHOLD,
    DB_PATH,
    GAP_HIGH,
    GAP_LOW,
    NULL_PCT_THRESHOLD,
    ORDERS_TABLE,
    POLLING_INTERVAL_MINUTES,
    Z_SCORE_THRESHOLD,
)
from logging_config import logger
from state import PipelineState


class MonitorAgent:
    """Monitors SQL data pipelines for anomalies.

    Runs every 5 minutes, computes statistical baselines from historical data,
    and detects three anomaly types:
    - missing_data: row count significantly below baseline
    - data_quality: null rate above threshold
    - schema_drift: (added in Phase 5)
    """

    def __init__(self, db_path: str = None) -> None:
        """Initialize MonitorAgent.

        Args:
            db_path: Path to the SQLite database. Defaults to config.DB_PATH.
        """
        if db_path is None:
            import config as _cfg
            db_path = _cfg.DB_PATH
        self.db_path = db_path

    def _get_connection(self) -> DBWrapper:
        """Get an abstracted database connection."""
        return get_db_connection(self.db_path)

    def _get_current_count(self, conn: DBWrapper, now: datetime) -> int:
        """Get row count for the most recent 5-minute window.

        Args:
            conn: SQLite connection.
            now: Current UTC timestamp.

        Returns:
            Number of rows in the current window.
        """
        window_start = now - timedelta(minutes=POLLING_INTERVAL_MINUTES)
        result = conn.execute(
            f"SELECT COUNT(*) FROM {ORDERS_TABLE} WHERE created_at > ?",
            (window_start.isoformat(),),
        ).fetchone()
        return result[0] if result else 0

    def _compute_baseline(self, conn: DBWrapper, now: datetime) -> Dict[str, float]:
        """Compute 7-day rolling average and stddev of rows per 5-min window using pandas.

        Queries the last 7 days of data, groups into 5-min windows,
        and returns the mean and standard deviation of row counts per window.

        Args:
            conn: SQLite connection.
            now: Current UTC timestamp.

        Returns:
            Dict containing 'mean' and 'std' for row counts.
        """
        baseline_start = now - timedelta(days=7)

        query = f"""
            SELECT created_at FROM {ORDERS_TABLE}
            WHERE created_at > ? AND created_at <= ?
        """
        
        converted_query = conn._convert_sql(query)

        df = pd.read_sql_query(
            converted_query,
            conn.conn,
            params=(baseline_start.isoformat(), now.isoformat()),
        )

        if df.empty:
            logger.warning("No baseline data found for the last 7 days")
            return {"mean": 0.0, "std": 0.0}

        df["created_at"] = pd.to_datetime(df["created_at"], format="ISO8601")

        # Group into 5-minute windows and count rows per window
        df["window"] = df["created_at"].dt.floor("5min")
        window_counts = df.groupby("window").size()

        mean_val = float(window_counts.mean())
        std_val = float(window_counts.std()) if len(window_counts) > 1 else 0.0
        
        logger.info(
            f"Baseline: {mean_val:.1f} rows/window, std: {std_val:.1f} "
            f"(from {len(window_counts)} windows)"
        )

        return {"mean": mean_val, "std": std_val}

    def _check_null_rate(self, conn: DBWrapper, now: datetime) -> float:
        """Calculate null rate on order_amount for the recent window.

        Args:
            conn: SQLite connection.
            now: Current UTC timestamp.

        Returns:
            Fraction of rows with null order_amount (0.0 to 1.0).
        """
        window_start = now - timedelta(minutes=POLLING_INTERVAL_MINUTES)

        result = conn.execute(
            f"""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN order_amount IS NULL THEN 1 ELSE 0 END) as nulls
            FROM {ORDERS_TABLE}
            WHERE created_at > ?
            """,
            (window_start.isoformat(),),
        ).fetchone()

        total = result[0] if result[0] else 0
        nulls = result[1] if result[1] else 0

        if total == 0:
            return 0.0

        return nulls / total

    def _estimate_gap_minutes(self, conn: DBWrapper, now: datetime) -> float:
        """Estimate how long data has been missing.

        Finds the timestamp of the most recent order and calculates
        the gap between that and now.

        Args:
            conn: SQLite connection.
            now: Current UTC timestamp.

        Returns:
            Gap duration in minutes, or 0.0 if no significant gap.
        """
        result = conn.execute(
            f"SELECT MAX(created_at) FROM {ORDERS_TABLE}"
        ).fetchone()

        if not result or not result[0]:
            return 0.0

        last_record = datetime.fromisoformat(str(result[0]))
        gap = (now - last_record).total_seconds() / 60.0

        # Only report gap if it exceeds the polling interval
        if gap > POLLING_INTERVAL_MINUTES * 2:
            return gap
        return 0.0

    def _assign_severity(self, gap_minutes: float, null_rate: float) -> str:
        """Assign severity based on gap duration and null rate.

        Severity rules (from project spec):
        - gap < 30 min OR null rate < 5%: LOW
        - gap 30 min to 6 hours: MEDIUM
        - gap > 6 hours OR null rate > 5%: HIGH

        Args:
            gap_minutes: Duration of detected data gap.
            null_rate: Fraction of null values in critical column.

        Returns:
            Severity string: 'LOW', 'MEDIUM', 'HIGH', or 'NONE'.
        """
        if null_rate > NULL_PCT_THRESHOLD:
            return "HIGH"

        if gap_minutes <= 0:
            if null_rate > 0:
                return "LOW"
            return "NONE"

        if gap_minutes < GAP_LOW:
            return "LOW"
        elif gap_minutes <= GAP_HIGH:
            return "MEDIUM"
        else:
            return "HIGH"

    def run(self, state: PipelineState) -> PipelineState:
        """Execute the monitoring check and update state using dbt.
        
        If the state already has an anomaly (e.g. from the webhook Dead Man's Switch),
        we skip the dbt checks and pass it along.
        """
        run_id = state["run_id"]
        logger.info(f"[{run_id}] Monitor Agent starting dbt check...")

        if state.get("anomaly_detected"):
            logger.info(f"[{run_id}] Anomaly already detected by Webhook: {state['anomaly_type']}. Skipping dbt.")
            return state

        # Run dbt test
        import subprocess
        import json
        import os
        
        project_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dbt_pipeline")
        logger.info(f"[{run_id}] Running dbt test in {project_dir}...")
        
        try:
            result = subprocess.run(
                ["dbt", "test", "--profiles-dir", "."], 
                cwd=project_dir, 
                capture_output=True, 
                text=True
            )
            
            run_results_path = os.path.join(project_dir, "target", "run_results.json")
            if not os.path.exists(run_results_path):
                logger.error(f"[{run_id}] dbt run_results.json not found!")
                return state
                
            with open(run_results_path, "r") as f:
                data = json.load(f)
                
            failures = [r for r in data.get("results", []) if r.get("status") == "fail"]
            
            anomaly_detected = False
            anomaly_type = ""
            severity = "NONE"
            affected_tables = []
            
            if failures:
                anomaly_detected = True
                affected_tables = ["orders"]
                
                # Check what kind of test failed
                failure_names = [f.get("unique_id", "") for f in failures]
                if any("unique" in name for name in failure_names):
                    anomaly_type = "data_duplication"
                    severity = "MEDIUM"
                elif any("not_null" in name for name in failure_names):
                    anomaly_type = "data_quality"
                    severity = "HIGH"
                elif any("relationships" in name for name in failure_names):
                    anomaly_type = "referential_integrity"
                    severity = "MEDIUM"
                else:
                    anomaly_type = "data_quality"
                    severity = "LOW"
                    
                logger.warning(f"[{run_id}] dbt test failures detected: {failure_names}")
                
            state["anomaly_detected"] = anomaly_detected
            state["anomaly_type"] = anomaly_type
            state["severity"] = severity
            state["affected_tables"] = affected_tables
            state["raw_count"] = 0
            state["expected_avg"] = 0.0
            state["z_score"] = 0.0
            state["null_rate"] = 0.0
            
            if anomaly_detected:
                logger.warning(
                    f"[{run_id}] dbt Anomaly detected: type={anomaly_type}, "
                    f"severity={severity}"
                )
            else:
                logger.info(f"[{run_id}] ✅ dbt tests passed. Pipeline healthy.")
                
        except Exception as e:
            logger.error(f"[{run_id}] dbt execution failed: {e}")
            
        return state
