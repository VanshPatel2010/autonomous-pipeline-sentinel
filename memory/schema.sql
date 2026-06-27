-- Incident log: episodic long-term memory (LTM)
-- Persists across runs. Queried by Diagnoser for context.
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

-- Playbooks: procedural LTM (Phase 3)
CREATE TABLE IF NOT EXISTS playbooks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    anomaly_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    action_taken TEXT NOT NULL,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    last_used TEXT,
    UNIQUE(anomaly_type, severity, action_taken)
);

-- Schema snapshots: semantic LTM audit trail (Phase 5)
CREATE TABLE IF NOT EXISTS schema_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name TEXT NOT NULL,
    schema_json TEXT NOT NULL,
    snapshot_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Data gaps: gap tracking (Phase 3)
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
