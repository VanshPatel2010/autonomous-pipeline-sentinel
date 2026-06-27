"""Monitor Agent: polls SQLite for pipeline health every 5 minutes.

Checks:
- Row count vs 7-day rolling baseline
- Null rate on critical columns (order_amount)
- Data staleness (freshness window)
- Gap duration estimation

Writes findings into the LangGraph state dict (STM).
No LLM calls — purely statistical anomaly detection.
"""

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import pandas as pd

from config import (
    ANOMALY_THRESHOLD,
    DB_PATH,
    GAP_HIGH,
    GAP_LOW,
    NULL_PCT_THRESHOLD,
    ORDERS_TABLE,
    POLLING_INTERVAL_MINUTES,
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

    def _get_connection(self) -> sqlite3.Connection:
        """Get a SQLite connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _get_current_count(self, conn: sqlite3.Connection, now: datetime) -> int:
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

    def _compute_baseline(self, conn: sqlite3.Connection, now: datetime) -> float:
        """Compute 7-day rolling average of rows per 5-min window using pandas.

        Queries the last 7 days of data, groups into 5-min windows,
        and returns the average row count per window.

        Args:
            conn: SQLite connection.
            now: Current UTC timestamp.

        Returns:
            Average row count per 5-min window over the last 7 days.
        """
        baseline_start = now - timedelta(days=7)

        query = f"""
            SELECT created_at FROM {ORDERS_TABLE}
            WHERE created_at > ? AND created_at <= ?
        """

        df = pd.read_sql_query(
            query,
            conn,
            params=(baseline_start.isoformat(), now.isoformat()),
        )

        if df.empty:
            logger.warning("No baseline data found for the last 7 days")
            return 0.0

        df["created_at"] = pd.to_datetime(df["created_at"], format="ISO8601")

        # Group into 5-minute windows and count rows per window
        df["window"] = df["created_at"].dt.floor("5min")
        window_counts = df.groupby("window").size()

        baseline_avg = float(window_counts.mean())
        logger.info(
            f"Baseline: {baseline_avg:.1f} rows/window "
            f"(from {len(window_counts)} windows)"
        )

        return baseline_avg

    def _check_null_rate(self, conn: sqlite3.Connection, now: datetime) -> float:
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

    def _estimate_gap_minutes(self, conn: sqlite3.Connection, now: datetime) -> float:
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
        """Execute the monitoring check and update state.

        Steps:
        1. Query current 5-min window row count
        2. Compute 7-day rolling average baseline
        3. Check null rate on order_amount
        4. Estimate gap duration if anomaly found
        5. Assign severity and update state

        Args:
            state: Current pipeline state dict.

        Returns:
            Updated state dict with monitor findings.
        """
        run_id = state["run_id"]
        logger.info(f"[{run_id}] Monitor Agent starting check...")

        conn = self._get_connection()
        now = datetime.now(timezone.utc).replace(tzinfo=None)

        try:
            # Step 0: Check for schema drift (Phase 5)
            try:
                from memory.schema_registry import check_drift
                drift_result = check_drift(conn, 'orders')
                if drift_result['drift_detected']:
                    logger.warning(
                        f'[{run_id}] SCHEMA DRIFT DETECTED: '
                        f'new={drift_result["new_columns"]}, '
                        f'deleted={drift_result["deleted_columns"]}, '
                        f'type_changes={drift_result["type_changes"]}'
                    )
                    state['anomaly_detected'] = True
                    state['anomaly_type'] = 'schema_drift'
                    state['severity'] = 'MEDIUM'
                    state['gap_minutes'] = 0.0
                    state['affected_tables'] = ['orders']
                    state['raw_count'] = 0
                    state['expected_avg'] = 0.0
                    state['null_rate'] = 0.0
                    # Pre-fill diagnoser output (schema drift is deterministic)
                    state['diagnoser_output'] = {
                        'root_cause': f'Schema drift detected: {drift_result}',
                        'confidence': 1.0,
                        'estimated_missing_rows': 0,
                        'severity_override': 'MEDIUM',
                    }
                    conn.close()
                    return state
            except Exception as e:
                logger.debug(f'[{run_id}] Schema drift check skipped: {e}')

            # Step 1: Get current window count
            raw_count = self._get_current_count(conn, now)
            state["raw_count"] = raw_count
            logger.info(f"[{run_id}] Current window count: {raw_count}")

            # Step 2: Compute baseline
            baseline_avg = self._compute_baseline(conn, now)
            state["expected_avg"] = baseline_avg

            # Step 3: Check null rate
            null_rate = self._check_null_rate(conn, now)
            state["null_rate"] = null_rate
            logger.info(f"[{run_id}] Null rate: {null_rate:.2%}")

            # Step 4: Detect anomalies
            anomaly_detected = False
            anomaly_type = ""
            gap_minutes = 0.0

            # Check row count anomaly
            if baseline_avg > 0 and raw_count < (baseline_avg * ANOMALY_THRESHOLD):
                anomaly_detected = True
                anomaly_type = "missing_data"
                gap_minutes = self._estimate_gap_minutes(conn, now)
                logger.warning(
                    f"[{run_id}] ANOMALY: Row count {raw_count} < "
                    f"{ANOMALY_THRESHOLD:.0%} of baseline {baseline_avg:.0f}"
                )

            # Check null rate anomaly (overrides if worse)
            if null_rate > NULL_PCT_THRESHOLD:
                anomaly_detected = True
                anomaly_type = "data_quality"
                logger.warning(
                    f"[{run_id}] ANOMALY: Null rate {null_rate:.2%} > "
                    f"threshold {NULL_PCT_THRESHOLD:.2%}"
                )

            # Step 5: Assign severity
            severity = (
                self._assign_severity(gap_minutes, null_rate)
                if anomaly_detected
                else "NONE"
            )

            # Update state
            state["anomaly_detected"] = anomaly_detected
            state["anomaly_type"] = anomaly_type
            state["severity"] = severity
            state["gap_minutes"] = gap_minutes
            state["affected_tables"] = [ORDERS_TABLE] if anomaly_detected else []

            if anomaly_detected:
                logger.warning(
                    f"[{run_id}] Anomaly detected: type={anomaly_type}, "
                    f"severity={severity}, gap={gap_minutes:.0f}min"
                )
            else:
                logger.info(f"[{run_id}] No anomaly detected. Pipeline healthy.")

        finally:
            conn.close()

        return state
