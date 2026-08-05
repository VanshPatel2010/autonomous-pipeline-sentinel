"""Initialize PostgreSQL schemas for agent memory and simulator state."""

import logging
from db.client import get_db_connection

logger = logging.getLogger(__name__)

def setup_postgres_schemas() -> None:
    """Create all required agent memory tables in PostgreSQL if they don't exist."""
    conn = get_db_connection()
    if not conn.is_postgres:
        logger.info("Not using PostgreSQL, skipping postgres schema setup.")
        return

    logger.info("Setting up PostgreSQL schemas for agent memory...")

    # Episodic LTM
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS incidents (
            run_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            anomaly_type TEXT,
            severity TEXT,
            gap_minutes REAL DEFAULT 0,
            root_cause TEXT,
            affected_tables TEXT,       -- JSON array string
            fix_taken TEXT,
            resolved INTEGER DEFAULT 0,
            resolved_at TEXT,
            confidence REAL DEFAULT 0.0
        );
        """
    )

    # Procedural LTM
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS playbooks (
            id SERIAL PRIMARY KEY,
            anomaly_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            action_taken TEXT NOT NULL,
            success_count INTEGER DEFAULT 0,
            failure_count INTEGER DEFAULT 0,
            last_used TEXT,
            UNIQUE(anomaly_type, severity, action_taken)
        );
        """
    )

    # Semantic LTM
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_snapshots (
            id SERIAL PRIMARY KEY,
            table_name TEXT NOT NULL,
            schema_json TEXT NOT NULL,
            snapshot_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    # Gap Tracker
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS data_gaps (
            gap_id TEXT PRIMARY KEY,
            run_id TEXT,
            start_time TEXT,
            end_time TEXT,
            estimated_rows INTEGER,
            source_db TEXT,
            reconciled INTEGER DEFAULT 0,
            created_at TEXT
        );
        """
    )
    
    # Metrics Tracker
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS metrics_history (
            id SERIAL PRIMARY KEY,
            timestamp TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            tags TEXT  -- JSON dict of tags
        );
        """
    )
    
    # Repair Agent Memory Log
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_memory_log (
            id SERIAL PRIMARY KEY,
            run_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            memory_type TEXT NOT NULL, -- 'episodic', 'procedural', 'semantic'
            query TEXT NOT NULL,
            result TEXT,
            confidence REAL
        );
        """
    )
    
    # Advanced Memory Tiers
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS episodic_memory (
            id              SERIAL PRIMARY KEY,
            memory_id       TEXT NOT NULL UNIQUE,
            run_id          TEXT NOT NULL,
            anomaly_type    TEXT NOT NULL,
            severity        TEXT NOT NULL,
            root_cause      TEXT,
            strategy_used   TEXT,
            outcome         TEXT,
            verification_score REAL DEFAULT 0.0,
            recovery_secs   REAL DEFAULT 0.0,
            embedding_text  TEXT,
            created_at      TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_episodic_anomaly
            ON episodic_memory (anomaly_type, severity);
        CREATE INDEX IF NOT EXISTS idx_episodic_strategy
            ON episodic_memory (strategy_used, outcome);

        CREATE TABLE IF NOT EXISTS semantic_memory (
            id          SERIAL PRIMARY KEY,
            memory_id   TEXT NOT NULL UNIQUE,
            concept     TEXT NOT NULL UNIQUE,
            knowledge   TEXT NOT NULL,
            confidence  REAL DEFAULT 0.5,
            source      TEXT DEFAULT 'learner',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_semantic_concept
            ON semantic_memory (concept);

        CREATE TABLE IF NOT EXISTS procedural_memory (
            id                      SERIAL PRIMARY KEY,
            memory_id               TEXT NOT NULL UNIQUE,
            procedure_name          TEXT NOT NULL,
            anomaly_type            TEXT NOT NULL,
            severity                TEXT NOT NULL,
            steps                   TEXT DEFAULT '[]',
            success_count           INTEGER DEFAULT 0,
            total_count             INTEGER DEFAULT 0,
            avg_verification_score  REAL DEFAULT 0.0,
            created_at              TEXT NOT NULL,
            updated_at              TEXT NOT NULL,
            UNIQUE (procedure_name, anomaly_type, severity)
        );
        CREATE INDEX IF NOT EXISTS idx_procedural_lookup
            ON procedural_memory (anomaly_type, severity);

        CREATE TABLE IF NOT EXISTS self_reflections (
            id                  SERIAL PRIMARY KEY,
            run_id              TEXT NOT NULL,
            plan_id             TEXT,
            strategy_chosen     TEXT,
            verification_passed INTEGER DEFAULT 0,
            verification_score  REAL DEFAULT 0.0,
            optimal_choice      INTEGER DEFAULT 1,
            cheaper_alternative TEXT,
            downtime_reduction  REAL DEFAULT 0.0,
            confidence_adjustment REAL DEFAULT 0.0,
            learning_points     TEXT DEFAULT '[]',
            reflection_text     TEXT,
            created_at          TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_reflections_run
            ON self_reflections (run_id);
            
        CREATE TABLE IF NOT EXISTS repair_memory (
            id                  SERIAL PRIMARY KEY,
            run_id              TEXT NOT NULL,
            anomaly_type        TEXT NOT NULL,
            severity            TEXT NOT NULL,
            root_cause          TEXT,
            repair_plan_id      TEXT,
            strategy_name       TEXT NOT NULL,
            execution_time_secs REAL DEFAULT 0,
            recovery_time_secs  INTEGER DEFAULT 0,
            affected_rows       INTEGER DEFAULT 0,
            node_used           TEXT,
            repair_confidence   REAL DEFAULT 0.5,
            business_impact     REAL DEFAULT 0.0,
            failure_reason      TEXT,
            verification_score  REAL DEFAULT 0.0,
            operator_feedback   TEXT,
            final_outcome       TEXT NOT NULL DEFAULT 'unknown',
            utility_score       REAL DEFAULT 0.0,
            hypothesis_rank     INTEGER DEFAULT 1,
            llm_assisted        INTEGER DEFAULT 0,
            step_count          INTEGER DEFAULT 0,
            pre_check_passed    INTEGER DEFAULT 0,
            error_encountered   INTEGER DEFAULT 0,
            created_at          TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_repair_memory_anomaly
            ON repair_memory (anomaly_type, severity);

        CREATE INDEX IF NOT EXISTS idx_repair_memory_strategy
            ON repair_memory (strategy_name, final_outcome);
        """
    )

    # Simulator State Config
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS simulator_config (
            id INTEGER PRIMARY KEY,
            inject_gap BOOLEAN DEFAULT FALSE,
            inject_nulls BOOLEAN DEFAULT FALSE,
            inject_duplicate BOOLEAN DEFAULT FALSE
        );
        """
    )

    # Ensure single row exists for simulator config
    conn.execute(
        """
        INSERT INTO simulator_config (id, inject_gap, inject_nulls, inject_duplicate)
        VALUES (1, FALSE, FALSE, FALSE)
        ON CONFLICT (id) DO NOTHING;
        """
    )

    conn.commit()
    conn.close()
    logger.info("PostgreSQL agent memory schemas initialized successfully.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    setup_postgres_schemas()
