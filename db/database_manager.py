"""DatabaseManager: routes all pipeline DB connections through PRIMARY → REPLICA chain.

Simulates real database failover. State is persisted to db/failover_state.json
so the dashboard always knows which node is active.
"""

import json
import os
import shutil
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List

from logging_config import logger

# ── project root ──────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from config import DB_PATH

# Base directory is the same folder as pipeline.db
_base_dir = os.path.dirname(os.path.abspath(DB_PATH))

DB_NODES: List[Dict[str, Any]] = [
    {
        "id": "primary",
        "label": "PRIMARY",
        "path": os.path.abspath(DB_PATH),   # ← Same as pipeline.db, not data/primary/orders.db
        "port": 8080,
        "role": "primary",
        "color": "#238636",
    },
    {
        "id": "replica1",
        "label": "REPLICA-1",
        "path": os.path.join(_base_dir, "data", "replica1", "orders.db"),
        "port": 8081,
        "role": "replica",
        "color": "#1f6feb",
    },
    {
        "id": "replica2",
        "label": "REPLICA-2",
        "path": os.path.join(_base_dir, "data", "replica2", "orders.db"),
        "port": 8082,
        "role": "replica",
        "color": "#8957e5",
    },
]

FAILOVER_STATE_PATH = os.path.join(_PROJECT_ROOT, "db", "failover_state.json")
FAILOVER_LOG_PATH = os.path.join(_PROJECT_ROOT, "db", "failover_log.json")


# ═══════════════════════════════════════════════════════════════════════════
# STATE PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════

def _load_state() -> dict:
    """Load failover state from disk."""
    if os.path.exists(FAILOVER_STATE_PATH):
        try:
            with open(FAILOVER_STATE_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {
        "active_node_id": "primary",
        "failed_nodes": [],
        "failover_count": 0,
        "last_failover": None,
    }


def _save_state(state: dict) -> None:
    """Persist failover state to disk."""
    os.makedirs(os.path.dirname(FAILOVER_STATE_PATH), exist_ok=True)
    with open(FAILOVER_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def _load_failover_log() -> list:
    """Load the failover event log."""
    if os.path.exists(FAILOVER_LOG_PATH):
        try:
            with open(FAILOVER_LOG_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return []


def _append_failover_log(entry: dict) -> None:
    """Append a failover event and keep last 50 entries."""
    log = _load_failover_log()
    log.append(entry)
    os.makedirs(os.path.dirname(FAILOVER_LOG_PATH), exist_ok=True)
    with open(FAILOVER_LOG_PATH, "w") as f:
        json.dump(log[-50:], f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# NODE QUERIES
# ═══════════════════════════════════════════════════════════════════════════

def get_active_node() -> dict:
    """Return the currently active DB node dict."""
    state = _load_state()
    for node in DB_NODES:
        if node["id"] == state["active_node_id"]:
            return node
    return DB_NODES[0]


def get_active_db_path() -> str:
    """Return the SQLite file path for the active node."""
    return get_active_node()["path"]


def get_all_nodes_status() -> list:
    """Return all nodes with their current status (active/standby/failed)."""
    state = _load_state()
    result = []
    for node in DB_NODES:
        status = "standby"
        if node["id"] == state["active_node_id"]:
            status = "active"
        elif node["id"] in state.get("failed_nodes", []):
            status = "failed"
        result.append({**node, "status": status})
    return result


def get_failover_state() -> dict:
    """Return the raw failover state dict."""
    return _load_state()


def get_node_row_count(node: dict) -> int:
    """Query the row count in a specific node's orders table."""
    try:
        if not os.path.exists(node["path"]):
            return 0
        conn = sqlite3.connect(node["path"])
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='orders'"
        )
        if cur.fetchone() is None:
            conn.close()
            return 0
        count = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0


# ═══════════════════════════════════════════════════════════════════════════
# CONNECTION
# ═══════════════════════════════════════════════════════════════════════════

def get_conn() -> sqlite3.Connection:
    """Get a connection to the currently active database node."""
    path = get_active_db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return sqlite3.connect(path)


# ═══════════════════════════════════════════════════════════════════════════
# FAILOVER
# ═══════════════════════════════════════════════════════════════════════════

def trigger_failover(reason: str, severity: str) -> dict:
    """Switch to the next available replica node.

    Called by the Repairer agent for MEDIUM/HIGH/CRITICAL anomalies.

    Args:
        reason: Why the failover was triggered (e.g. root cause).
        severity: The anomaly severity level.

    Returns:
        Dict describing the failover result.
    """
    state = _load_state()
    current_node = get_active_node()

    # Mark current as failed
    failed = set(state.get("failed_nodes", []))
    failed.add(current_node["id"])

    # Find next non-failed node
    next_node = None
    for node in DB_NODES:
        if node["id"] not in failed:
            next_node = node
            break

    if next_node is None:
        logger.critical(
            "ALL database nodes have failed! Manual intervention required."
        )
        return {
            "success": False,
            "error": "ALL_NODES_FAILED",
            "message": "No healthy replica available. Manual intervention required.",
            "from_node": current_node,
            "to_node": None,
        }

    # STEP 1: Copy data FIRST, before updating state
    # This prevents race condition where dashboard reads new state before file exists
    os.makedirs(os.path.dirname(next_node["path"]), exist_ok=True)
    copy_success = False
    if os.path.exists(current_node["path"]):
        try:
            shutil.copy2(current_node["path"], next_node["path"])
            copy_success = True
            logger.info(
                f"Data synced: {current_node['label']} -> {next_node['label']}"
            )
        except Exception as e:
            logger.warning(f"DB copy failed: {e}")

    # STEP 2: Only THEN update state
    now = datetime.now(timezone.utc).isoformat()
    state["failed_nodes"] = list(failed)
    state["active_node_id"] = next_node["id"]
    state["failover_count"] = state.get("failover_count", 0) + 1
    state["last_failover"] = now
    _save_state(state)

    # STEP 3: Log
    log_entry = {
        "timestamp": now,
        "from_node": current_node["id"],
        "to_node": next_node["id"],
        "reason": reason,
        "severity": severity,
        "failover_count": state["failover_count"],
        "data_copied": copy_success,
    }
    _append_failover_log(log_entry)

    logger.warning(
        f"FAILOVER: {current_node['label']} -> {next_node['label']} "
        f"(reason: {reason}, severity: {severity})"
    )

    return {
        "success": True,
        "from_node": current_node,
        "to_node": next_node,
        "message": f"Failover complete: {current_node['label']} -> {next_node['label']}",
        "log_entry": log_entry,
        "data_copied": copy_success,
    }


def reset_all_nodes() -> None:
    """Reset cluster to PRIMARY, clear failed list.

    Used by the dashboard 'Reset Cluster' button and by seed_database().
    Copies the primary DB to all replicas so they're in sync.
    """
    state = {
        "active_node_id": "primary",
        "failed_nodes": [],
        "failover_count": 0,
        "last_failover": None,
    }
    _save_state(state)

    # Sync primary to replicas
    primary_path = DB_NODES[0]["path"]
    if os.path.exists(primary_path):
        for node in DB_NODES[1:]:
            try:
                os.makedirs(os.path.dirname(node["path"]), exist_ok=True)
                shutil.copy2(primary_path, node["path"])
            except Exception as e:
                logger.warning(f"Failed to sync {node['label']}: {e}")

    logger.info("Cluster reset: all nodes active, PRIMARY is active node")


def get_failover_log() -> list:
    """Return the failover event log."""
    return _load_failover_log()
