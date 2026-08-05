"""Central configuration for the Autonomous Data Pipeline Monitor.

All thresholds and constants live here. Agents import from config,
never hardcode values. Magic numbers in agent code are forbidden.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Database ---
DATABASE_URL: str = os.getenv("DATABASE_URL", "")
DB_PATH: str = os.getenv("DB_PATH", "pipeline.db")
BACKUP_TABLE: str = "backup_orders"
ORDERS_TABLE: str = "orders"

# --- Monitor Agent Thresholds ---
POLLING_INTERVAL_MINUTES: int = 5
BASELINE_WINDOW_DAYS: int = 7
BASELINE_WINDOWS: int = 2016  # 7 days * 24 hours * (60/5) windows per hour
ANOMALY_THRESHOLD: float = 0.4  # Alert if count < 40% of baseline
Z_SCORE_THRESHOLD: float = -3.0  # Alert if Z-score is < -3.0
NULL_PCT_THRESHOLD: float = 0.05  # Alert if null % > 5%
FRESHNESS_MINUTES: int = 10  # Alert if latest record older than this

# --- Severity Thresholds (gap duration in minutes) ---
GAP_LOW: int = 30  # < 30 min = LOW
GAP_HIGH: int = 360  # > 6 hours = HIGH, 30-360 = MEDIUM

# --- Diagnoser Agent ---
CONFIDENCE_MIN: float = 0.6  # Minimum confidence to proceed to repair
GROQ_MODEL: str = "llama-3.1-8b-instant"
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# --- Repairer Agent ---
MAX_RETRY_ATTEMPTS: int = 3
RETRY_WAIT_SECONDS: int = 300  # 5 minutes

# --- AI Reasoning Layer ---
# LLM provider for the LLMReasoner ('groq', 'mock', 'openai', 'claude', etc.)
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")

# Uncertainty threshold: decisions above this value trigger human escalation
UNCERTAINTY_THRESHOLD: float = float(os.getenv("UNCERTAINTY_THRESHOLD", "0.35"))

# Minimum memory samples before adaptive confidence overrides base prior
MEMORY_MIN_SAMPLES: int = int(os.getenv("MEMORY_MIN_SAMPLES", "3"))

# Procedural memory: minimum success rate to prefer stored procedure over template
PROCEDURAL_MIN_SUCCESS_RATE: float = float(os.getenv("PROCEDURAL_MIN_SUCCESS_RATE", "0.65"))

# Probabilistic forecast horizon in minutes
FORECAST_HORIZON_MINUTES: int = int(os.getenv("FORECAST_HORIZON_MINUTES", "60"))

# LLM reasoning: max tokens for hypothesis generation
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1200"))

# --- Slack ---
SLACK_WEBHOOK_URL: str = os.getenv("SLACK_WEBHOOK_URL", "")
SLACK_TIMEOUT: int = 10

# --- Mock Mode ---
# When True, skips Groq API and Slack webhook. All tests must run with MOCK_MODE=True.
MOCK_MODE: bool = os.getenv("MOCK_MODE", "True").lower() in ("true", "1", "yes")

# --- Data Simulation ---
SIMULATION_DAYS: int = 30
ROWS_PER_WINDOW: int = 200  # ~200 orders per 5-min window
NULL_RATE: float = 0.03  # 3% random nulls on order_amount
GAP_DURATION_HOURS: int = 4  # Simulated gap duration
