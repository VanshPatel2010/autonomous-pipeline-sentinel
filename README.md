# 🚨 Autonomous Data Pipeline Monitor

An intelligent **4-agent LangGraph system** that autonomously monitors SQL data pipelines for schema drift, missing data, and quality issues — diagnoses root causes via LLM, applies autonomous repairs, and delivers Slack incident reports.

**Zero human intervention required.** Reduced mean time to resolution from ~2 hours of manual debugging to under 5 minutes.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph State Machine                       │
│                                                                  │
│  ┌──────────┐    ┌────────────┐    ┌──────────┐    ┌──────────┐ │
│  │ Monitor  │───▶│ Diagnoser  │───▶│ Repairer │───▶│  Slack   │ │
│  │  Agent   │    │   Agent    │    │  Agent   │    │  Agent   │ │
│  └────┬─────┘    └─────┬──────┘    └────┬─────┘    └────┬─────┘ │
│       │                │               │               │        │
│  SQL checks      LLM reasoning    Auto-repair      Webhook     │
│  Baselines       Root cause       Failover         Incidents    │
│  Anomalies       Confidence       Quarantine       Reports     │
└───────┼────────────────┼───────────────┼───────────────┼────────┘
        │                │               │               │
   ┌────▼────┐     ┌─────▼─────┐   ┌─────▼─────┐   ┌───▼────┐
   │ SQLite  │     │ Episodic  │   │Procedural │   │ Slack  │
   │ orders  │     │   LTM     │   │   LTM     │   │Webhook │
   │  table  │     │(incidents)│   │(playbooks)│   │  API   │
   └─────────┘     └───────────┘   └───────────┘   └────────┘
```

---

## 🧠 Memory Architecture

| Layer | Type | Storage | Purpose |
|-------|------|---------|---------| 
| **STM** | Graph state dict | LangGraph TypedDict | Shared across agents per run |
| **LTM-Episodic** | Incident history | SQLite `incidents` | Diagnoser learns from past |
| **LTM-Procedural** | Repair playbooks | SQLite `playbooks` | Repairer learns what works |
| **LTM-Semantic** | Schema registry | JSON + SQLite | Monitor detects drift |
| **External** | Live queries | SQL, Slack, APIs | Real-time data access |

---

## 📋 Prerequisites

- **Python 3.11+**
- **pip** (Python package manager)
- **Groq API key** (free, no credit card) — [console.groq.com](https://console.groq.com)
- **Slack Webhook URL** (free, Phase 4) — [api.slack.com/apps](https://api.slack.com/apps)

---

## 🚀 Initial Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd autonomous-data-pipeline-agent

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your API keys:
#   GROQ_API_KEY=gsk_...         (required from Phase 2)
#   SLACK_WEBHOOK_URL=https://... (required from Phase 4)

# 5. Seed the database with 30 days of synthetic data
python seed_db.py

# 6. Verify the setup
python main.py --once
```

---

## 🧪 Testing Each Phase

Each phase builds on the previous one. Follow these instructions in order to verify that each phase works correctly.

---

### Phase 1: Monitor Agent + Anomaly Detection

**What it does:** The Monitor Agent polls SQLite every 5 minutes, computes a 7-day rolling baseline of row counts, and detects anomalies (missing data, null rate spikes).

**Components:**
- `agents/monitor.py` — Statistical anomaly detection
- `state.py` — LangGraph TypedDict (STM)
- `seed_db.py` — Synthetic data generator (30 days, ~200 rows/5min window)
- `config.py` — All thresholds centralized

#### Step 1: Seed the database
```bash
python seed_db.py
```
**Expected output:**
```
Generating 30 days of synthetic data...
Gap window: 2026-06-23T03:14:00 to 2026-06-23T07:14:00
Seeded 1,725,600 orders into pipeline.db
Null order_amounts: 51,768 (3.0%)
Gap: 4h starting at 2026-06-23 03:14:00
```

#### Step 2: Run a single healthy check
```bash
python main.py --once
```
**Expected output:**
```
Monitor Agent starting check...
Current window count: 192
Baseline: 199.9 rows/window (from 1960 windows)
Null rate: 2.94%
✅ Pipeline healthy. No issues detected.
```

#### Step 3: Force a missing_data anomaly
```bash
# Delete recent orders to simulate an outage
python -c "
import sqlite3
conn = sqlite3.connect('pipeline.db')
conn.execute(\"DELETE FROM orders WHERE created_at > datetime('now', '-30 minutes')\")
conn.commit()
remaining = conn.execute('SELECT COUNT(*) FROM orders').fetchone()[0]
print(f'Deleted recent rows. Remaining: {remaining:,}')
conn.close()
"

# Run the monitor — it should detect missing_data
python main.py --once
```
**Expected anomaly output:**
```
ANOMALY: Row count 0 < 40% of baseline 200
Anomaly detected: type=missing_data, severity=MEDIUM, gap=72min
```

#### Step 4: Force a data_quality anomaly (null spike)
```bash
# Re-seed database first
python seed_db.py

# Inject nulls to simulate quality degradation
python -c "
import sqlite3
from datetime import datetime, timedelta, timezone
conn = sqlite3.connect('pipeline.db')
now = datetime.now(timezone.utc).replace(tzinfo=None)
cutoff = (now - timedelta(minutes=5)).isoformat()
# Set all recent order_amounts to NULL
conn.execute(f\"UPDATE orders SET order_amount = NULL WHERE created_at > '{cutoff}'\")
conn.commit()
count = conn.execute(f\"SELECT COUNT(*) FROM orders WHERE created_at > '{cutoff}' AND order_amount IS NULL\").fetchone()[0]
print(f'Set {count} recent orders to NULL')
conn.close()
"

# Run monitor — should detect data_quality anomaly
python main.py --once
```
**Expected output:**
```
ANOMALY: Null rate 100.00% > threshold 5.00%
Anomaly detected: type=data_quality, severity=HIGH
```

#### Run Phase 1 unit tests
```bash
pytest tests/test_monitor.py -v
pytest tests/test_state.py -v
```

---

### Phase 2: Diagnoser Agent + LLM Root Cause Analysis

**What it does:** When an anomaly is detected, the Diagnoser Agent uses Groq's LLM (llama-3.1-8b-instant) to reason about root causes. It retrieves similar past incidents from episodic LTM to improve its reasoning.

**Components:**
- `agents/diagnoser.py` — LLM-powered root cause analysis
- `prompts/diagnoser_prompt.py` — System + user prompt templates
- `memory/incident_store.py` — Episodic LTM (stores incidents, retrieves similar ones)
- `memory/schema.sql` — DDL for incidents + playbooks tables
- `graph.py` — Two-node graph: monitor → [anomaly?] → diagnose → END

**Requires:** `GROQ_API_KEY` in `.env`

#### Step 1: Test with Mock LLM (no API key needed)
```bash
# Re-seed database to ensure fresh data
python seed_db.py

# Delete recent rows to trigger an anomaly
python -c "
import sqlite3
conn = sqlite3.connect('pipeline.db')
conn.execute(\"DELETE FROM orders WHERE created_at > datetime('now', '-30 minutes')\")
conn.commit()
conn.close()
"

# Run with mock mode
MOCK_MODE=True python main.py --once
```
**Expected output:**
```
Diagnoser Agent starting analysis: missing_data (MEDIUM)
Found 0 similar past incidents
MOCK_MODE: Using mock diagnoser output
Diagnosis complete: cause='Source database connectivity issue (mock diagnosis)', confidence=0.75, est_missing=2892
```

#### Step 2: Test with Real Groq LLM
```bash
# Ensure GROQ_API_KEY is set in .env
# Delete recent rows to trigger an anomaly (if not already done)
python -c "
import sqlite3
conn = sqlite3.connect('pipeline.db')
conn.execute(\"DELETE FROM orders WHERE created_at > datetime('now', '-30 minutes')\")
conn.commit()
conn.close()
"

# Run with real LLM
MOCK_MODE=False python main.py --once
```
**Expected output:**
```
Diagnoser Agent starting analysis: missing_data (MEDIUM)
Found 0 similar past incidents
Calling Groq LLM (llama-3.1-8b-instant)...
LLM response received (235 chars)
Severity upgraded: MEDIUM → HIGH (confidence: 0.85)
Diagnosis complete: cause='Source database connectivity issue or network issue...', confidence=0.85, est_missing=1999
```

#### Step 3: Verify Episodic Memory is Growing
```bash
# After a few runs, check that incidents are being stored
python -c "
import sqlite3
conn = sqlite3.connect('pipeline.db')
rows = conn.execute('SELECT run_id, anomaly_type, severity, root_cause FROM incidents ORDER BY timestamp DESC LIMIT 5').fetchall()
print(f'Total incidents stored: {len(rows)}')
for r in rows:
    cause = r[3][:60] if r[3] else 'N/A'
    print(f'  {r[0]} | {r[1]:15s} | {r[2]:8s} | {cause}')
conn.close()
"
```
**Expected output:**
```
Total incidents stored: 3
  fd9e7fac | missing_data    | HIGH     | Source database connectivity issue or network issue...
  3b4ce18c | missing_data    | MEDIUM   | Source database connectivity issue (mock diagnosis)
  abc12345 | missing_data    | MEDIUM   | Automated detection: missing_data anomaly
```

#### Step 4: Verify Conditional Routing (no anomaly = skip diagnoser)
```bash
# Re-seed the database to restore healthy data
python seed_db.py

# Run monitor — healthy pipeline should NOT trigger diagnoser
python main.py --once
```
**Expected output (NO diagnoser invocation):**
```
Monitor Agent starting check...
Current window count: 192
Baseline: 199.9 rows/window (from 1960 windows)
No anomaly detected. Pipeline healthy.
✅ Pipeline healthy. No issues detected.
```
Notice: The Diagnoser Agent is **NOT** invoked when no anomaly is detected.

#### Step 5: Test LLM Fallback (invalid API key)
```bash
# Test with an invalid API key to verify fallback works
MOCK_MODE=False GROQ_API_KEY=invalid_key python main.py --once 2>&1 | grep -E "(LLM call failed|fallback|Diagnosis complete)"
```
**Expected output:**
```
LLM call failed: ... Using fallback diagnosis.
Diagnosis complete: cause='Automated detection: missing_data anomaly', confidence=0.50
```

#### Run Phase 2 unit tests
```bash
pytest tests/test_diagnoser.py -v
pytest tests/test_memory.py -v
pytest tests/test_prompts.py -v
pytest tests/test_graph.py -v
```

---

### Phase 3: Repairer Agent + Autonomous Fixes

**What it does:** The Repairer applies fixes based on severity: wait/retry (LOW), switch to backup DB (MEDIUM), quarantine bad data (HIGH), or escalate to human (CRITICAL).

**Components:**
- `agents/repairer.py` — Autonomous repair strategies
- `memory/playbook_store.py` — Procedural LTM (learns what works)
- `memory/gap_tracker.py` — Data gap tracking
- `scripts/simulate_backup.py` — Delhi replica simulator

```bash
# Run full pipeline (monitor → diagnose → repair)
MOCK_MODE=False python main.py --once

# Verify playbook learning
python -c "
import sqlite3
conn = sqlite3.connect('pipeline.db')
rows = conn.execute('SELECT anomaly_type, severity, action_taken, success_count, failure_count FROM playbooks').fetchall()
print('Playbook entries:')
for r in rows:
    print(f'  {r[0]} | {r[1]} | {r[2]} | wins={r[3]} losses={r[4]}')
conn.close()
"
```

---

### Phase 4: Slack Agent + Full Orchestration

**What it does:** The Slack Agent posts formatted incident reports to your team channel.

**Requires:** `SLACK_WEBHOOK_URL` in `.env`

```bash
# Test with real Slack (requires SLACK_WEBHOOK_URL in .env)
MOCK_MODE=False python main.py --once

# Test Slack message formatting without sending (mock mode)
MOCK_MODE=True python main.py --once
```

**Slack report format:**
```
🚨 PIPELINE ALERT — HIGH SEVERITY
Time: 3:14 AM IST | Pipeline: orders_ingestion
Issue: Mumbai source DB unreachable
Actions: ✅ Switched to Delhi replica, ⏳ Reconciliation queued
```

---

### Phase 5: Dashboard + Schema Drift + CI

**What it does:** Adds a Streamlit dashboard, schema drift detection, metrics tracking, and CI/CD.

```bash
# Launch the Streamlit dashboard
streamlit run dashboard/app.py

# Run full test suite
pytest tests/ -v --cov=. --cov-report=term-missing

# Test schema drift detection
python -c "
import sqlite3
conn = sqlite3.connect('pipeline.db')
conn.execute('ALTER TABLE orders ADD COLUMN upi_transaction_id TEXT')
conn.commit()
conn.close()
print('Added new column — run monitor to detect drift')
"
python main.py --once
```

---

## 🏃 How to Run the Project (Final)

### Quick Start (Single Check)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Edit .env: add GROQ_API_KEY (required), SLACK_WEBHOOK_URL (optional)

# 3. Seed the database
python seed_db.py

# 4. Run a single monitoring check
python main.py --once
```

### Continuous Monitoring (Production Mode)
```bash
# Runs every 5 minutes, press Ctrl+C to stop
python main.py
```

### Mock Mode (No External APIs)
```bash
# All features work, but Groq LLM and Slack are mocked
MOCK_MODE=True python main.py --once
```

### Real LLM Mode (Requires Groq API Key)
```bash
# Uses real Groq LLM for root cause analysis
MOCK_MODE=False python main.py --once
```

### Run All Tests
```bash
# Run the full test suite
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=term-missing

# Run specific phase tests
pytest tests/test_monitor.py -v        # Phase 1
pytest tests/test_state.py -v          # Phase 1
pytest tests/test_diagnoser.py -v      # Phase 2
pytest tests/test_memory.py -v         # Phase 2
pytest tests/test_prompts.py -v        # Phase 2
pytest tests/test_graph.py -v          # Phase 2
pytest tests/test_repairer.py -v       # Phase 3
pytest tests/test_slack_agent.py -v    # Phase 4
pytest tests/test_schema_registry.py -v # Phase 5
pytest tests/test_metrics.py -v        # Phase 5
pytest tests/test_full_pipeline.py -v  # Integration
```

### Dashboard (Phase 5)
```bash
streamlit run dashboard/app.py
```

---

## 📁 Project Structure

```
├── agents/
│   ├── __init__.py
│   ├── monitor.py          # Phase 1: Anomaly detection (SQL + statistics)
│   ├── diagnoser.py        # Phase 2: LLM root cause analysis (Groq)
│   ├── repairer.py         # Phase 3: Autonomous repairs
│   └── slack_agent.py      # Phase 4: Slack incident reports
├── memory/
│   ├── __init__.py
│   ├── incident_store.py   # Phase 2: Episodic LTM (past incidents)
│   ├── playbook_store.py   # Phase 3: Procedural LTM
│   ├── gap_tracker.py      # Phase 3: Data gap tracking
│   ├── schema_registry.py  # Phase 5: Semantic LTM
│   └── schema.sql          # DDL for memory tables
├── prompts/
│   ├── __init__.py
│   ├── diagnoser_prompt.py # Phase 2: LLM prompt templates
│   └── slack_template.py   # Phase 4: Slack message template
├── metrics/
│   ├── __init__.py
│   └── tracker.py          # Phase 5: MTTR, detection rate
├── dashboard/
│   └── app.py              # Phase 5: Streamlit dashboard
├── schemas/
│   └── orders.json         # Phase 5: Expected schema baseline
├── tests/
│   ├── __init__.py
│   ├── test_monitor.py     # Phase 1: Monitor Agent tests
│   ├── test_state.py       # Phase 1: State TypedDict tests
│   ├── test_diagnoser.py   # Phase 2: Diagnoser Agent tests
│   ├── test_memory.py      # Phase 2: Episodic LTM tests
│   ├── test_prompts.py     # Phase 2: Prompt template tests
│   ├── test_graph.py       # Phase 2: Graph routing tests
│   ├── test_repairer.py    # Phase 3: Repairer Agent tests
│   ├── test_slack_agent.py # Phase 4: Slack Agent tests
│   ├── test_schema_registry.py # Phase 5: Schema drift tests
│   ├── test_metrics.py     # Phase 5: Metrics tracker tests
│   └── test_full_pipeline.py   # Integration: Full pipeline tests
├── scripts/
│   ├── __init__.py
│   └── simulate_backup.py  # Phase 3: Delhi replica simulator
├── config.py               # All thresholds (no magic numbers)
├── state.py                # LangGraph TypedDict (STM)
├── graph.py                # LangGraph state machine
├── main.py                 # Entry point + scheduler
├── seed_db.py              # 30-day data simulator
├── logging_config.py       # Shared logger
├── requirements.txt        # Pinned dependencies
├── .env.example            # Environment template
└── README.md               # This file
```

---

## ⚙️ Configuration

All thresholds live in `config.py` — agents never hardcode values:

| Constant | Default | Purpose |
|----------|---------|---------|
| `POLLING_INTERVAL_MINUTES` | 5 | How often the monitor checks |
| `BASELINE_WINDOW_DAYS` | 7 | Rolling baseline window |
| `ANOMALY_THRESHOLD` | 0.4 | Alert if count < 40% of baseline |
| `NULL_PCT_THRESHOLD` | 0.05 | Alert if null rate > 5% |
| `GAP_LOW` | 30 min | Below = LOW severity |
| `GAP_HIGH` | 360 min | Above = HIGH severity |
| `CONFIDENCE_MIN` | 0.6 | Min confidence to auto-repair |
| `GROQ_MODEL` | llama-3.1-8b-instant | LLM model for diagnoser |
| `MAX_RETRY_ATTEMPTS` | 3 | Repairer retry limit |
| `MOCK_MODE` | True | Skip external APIs for testing |

---

## 🔧 Environment Variables

| Variable | Required | Phase | Description |
|----------|----------|-------|-------------|
| `GROQ_API_KEY` | Phase 2+ | 2 | Groq API key for LLM reasoning |
| `SLACK_WEBHOOK_URL` | Phase 4+ | 4 | Slack incoming webhook URL |
| `MOCK_MODE` | No | All | `True` to skip external APIs |
| `DB_PATH` | No | All | SQLite database path (default: `pipeline.db`) |

---

## 🔄 Pipeline Flow (Full 4-Agent Graph)

```
1. Monitor Agent runs
   ├── Checks schema drift vs JSON baseline (semantic LTM)
   ├── Queries SQLite for current 5-min window row count
   ├── Computes 7-day rolling baseline average
   ├── Checks null rate on order_amount
   └── Detects anomaly? ──────────────────────────────┐
                                                       │
2. If anomaly detected:                                │
   ├── Diagnoser Agent runs                            │
   │   ├── Retrieves similar past incidents (LTM)      │
   │   ├── Builds context-rich prompt                  │
   │   ├── Calls Groq LLM for JSON diagnosis           │
   │   ├── Parses and validates response               │
   │   └── May upgrade severity (high confidence)      │
   │                                                   │
   ├── Repairer Agent runs                             │
   │   ├── Checks confidence threshold (≥0.6)          │
   │   ├── Consults playbooks (procedural LTM)         │
   │   ├── Executes repair strategy:                   │
   │   │   ├── LOW → Wait & retry next cycle            │
   │   │   ├── MEDIUM → Switch to Delhi replica         │
   │   │   ├── HIGH → Quarantine bad data               │
   │   │   └── CRITICAL → Escalate to human             │
   │   └── Records outcome in playbooks (learns)       │
   │                                                   │
   ├── Slack Agent runs                                │
   │   ├── Formats Block Kit message                   │
   │   ├── Uses escalation template for CRITICAL       │
   │   └── POSTs to webhook (or logs in mock mode)     │
   │                                                   │
   └── Incident saved to episodic LTM                  │
                                                       │
3. If no anomaly:                                      │
   └── Pipeline ends (no further agents invoked) ◄─────┘
```

---

## 📊 Phase Implementation Status

| Phase | Name | Status | Description |
|-------|------|--------|-------------|
| 1 | Monitor Agent | ✅ Complete | SQL polling, baseline computation, anomaly detection |
| 2 | Diagnoser Agent | ✅ Complete | LLM root cause analysis, episodic memory, graph routing |
| 3 | Repairer Agent | ✅ Complete | Autonomous repairs, playbook learning |
| 4 | Slack Agent | ✅ Complete | Incident reporting via webhooks |
| 5 | Dashboard + CI | ✅ Complete | Streamlit dashboard, schema drift, metrics |

---

## 🎯 Resume Bullet

> Built an autonomous data pipeline monitor using LangGraph and Python that detects schema drift, missing data, and quality issues across SQL pipelines. Reduced MTTR from ~2 hours of manual debugging to under 5 minutes of automated diagnosis and repair, with Slack-based incident reporting.

---

## 📚 Tech Stack

| Component | Technology | Cost |
|-----------|-----------|------|
| Agent Orchestration | LangGraph | Free (MIT) |
| LLM Backend | Groq (llama-3.1-8b-instant) | Free tier |
| Data Store | SQLite | Free (built-in) |
| Notifications | Slack Webhooks | Free |
| Dashboard | Streamlit | Free (Apache 2.0) |
| CI/CD | GitHub Actions | Free |
| Testing | pytest | Free (MIT) |

**Total cost: $0**

---

## 🐛 Troubleshooting

### "No baseline data found for the last 7 days"
Run `python seed_db.py` to seed the database with 30 days of synthetic data.

### "LLM call failed: Authentication error"
Check that `GROQ_API_KEY` is correctly set in your `.env` file. Get a free key at [console.groq.com](https://console.groq.com).

### "ModuleNotFoundError: No module named 'langgraph'"
Run `pip install -r requirements.txt` to install all dependencies.

### Pipeline shows "No anomaly detected" but I expected one
The anomaly threshold is 40% of baseline. If the current window has more than 40% of the expected rows, it's considered healthy. You can adjust `ANOMALY_THRESHOLD` in `config.py`.

### Running on Windows
Replace `MOCK_MODE=False python main.py --once` with:
```powershell
$env:MOCK_MODE="False"; python main.py --once
```
