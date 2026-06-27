"""Interactive Streamlit dashboard for the Autonomous Data Pipeline Monitor.

Replaces the passive log viewer with a full control center.
Launch with: streamlit run dashboard/app.py

Tabs:
  1. Overview   — KPIs, pipeline status, charts, recent incidents
  2. Run Simulation — inject anomalies + run the pipeline step-by-step
  3. Agent Intelligence — episodic memory, playbooks, schema registry
  4. Slack Preview — preview & send Slack alerts
"""

import json
import os
import sys
import time
import sqlite3
import uuid
import traceback
from datetime import datetime, timedelta, timezone

# ── project root on sys.path ──────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from config import DB_PATH, MOCK_MODE
from dotenv import set_key as dotenv_set_key

# ═════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Pipeline Monitor — Control Center",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS
# ═════════════════════════════════════════════════════════════════════════════

BG          = "#0d1117"
CARD_BG     = "#161b22"
BORDER      = "#30363d"
ACCENT      = "#1f6feb"
SUCCESS     = "#238636"
WARNING     = "#d29922"
DANGER      = "#da3633"
TXT         = "#f0f6fc"
TXT_MUTED   = "#8b949e"
NONE_GRAY   = "#8b949e"

SEVERITY_COLOR = {
    "CRITICAL": DANGER,
    "HIGH": DANGER,
    "MEDIUM": WARNING,
    "LOW": ACCENT,
    "NONE": NONE_GRAY,
}
SEVERITY_EMOJI = {
    "CRITICAL": "🔴",
    "HIGH": "🔴",
    "MEDIUM": "🟡",
    "LOW": "🟢",
    "NONE": "⚪",
}

PLOTLY_DARK = dict(
    paper_bgcolor='#161b22',
    plot_bgcolor='#0d1117',
    font=dict(color='#f0f6fc', family="monospace"),
    xaxis=dict(gridcolor='#30363d', zerolinecolor='#30363d'),
    yaxis=dict(gridcolor='#30363d', zerolinecolor='#30363d'),
    margin=dict(l=40, r=20, t=40, b=40),
)

# ═════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<style>
  /* global */
  .stApp {{ background-color: {BG}; }}
  [data-testid="stSidebar"] {{ background: linear-gradient(180deg,#0d1117,#161b22); border-right:1px solid {BORDER}; }}
  [data-testid="stMetric"] {{ background:{CARD_BG}; border:1px solid {BORDER}; border-radius:12px; padding:1rem 1.25rem; }}
  [data-testid="stMetricLabel"] {{ font-size:.8rem!important; text-transform:uppercase; letter-spacing:.06em; opacity:.7; }}
  [data-testid="stMetricValue"] {{ font-size:1.8rem!important; font-weight:700!important; }}
  .card {{ background:{CARD_BG}; border:1px solid {BORDER}; border-radius:12px; padding:1.2rem; margin-bottom:1rem; }}
  .banner-ok  {{ background:linear-gradient(135deg,{SUCCESS}22,{SUCCESS}11); border:1px solid {SUCCESS}55; border-radius:12px; padding:1.2rem 1.6rem; margin-bottom:1rem; }}
  .banner-bad {{ background:linear-gradient(135deg,{DANGER}22,{DANGER}11); border:1px solid {DANGER}55; border-radius:12px; padding:1.2rem 1.6rem; margin-bottom:1rem; }}
  .mono {{ font-family:monospace; font-size:.85rem; color:{TXT_MUTED}; }}
  .step-ok  {{ color:{SUCCESS}; }}
  .step-bad {{ color:{DANGER}; }}
  /* hide default header padding */
  .block-container {{ padding-top:1.5rem; }}
  /* slack preview card */
  .slack-card {{ background:#1a1d21; border-radius:8px; padding:1rem 1.2rem; color:#d1d2d3; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; font-size:.9rem; }}
  .slack-card h3 {{ color:#f2f3f5; margin:0 0 .6rem; font-size:1rem; }}
  .slack-field {{ display:inline-block; width:48%; vertical-align:top; margin-bottom:.5rem; }}
  .slack-field b {{ color:#f2f3f5; }}
  .slack-divider {{ border-top:1px solid #383a3e; margin:.7rem 0; }}
  .slack-ctx {{ color:#9ea1a5; font-size:.75rem; }}
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _db_path() -> str:
    """Resolve the absolute path to the pipeline database."""
    import config as _cfg
    p = _cfg.DB_PATH
    return p if os.path.isabs(p) else os.path.join(_PROJECT_ROOT, p)


def _db_exists() -> bool:
    return os.path.exists(_db_path())


def _table_exists(table: str) -> bool:
    if not _db_exists():
        return False
    try:
        conn = sqlite3.connect(_db_path())
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        )
        exists = cur.fetchone() is not None
        conn.close()
        return exists
    except Exception:
        return False


def _orders_count() -> int:
    if not _table_exists("orders"):
        return 0
    try:
        conn = sqlite3.connect(_db_path())
        n = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        conn.close()
        return n
    except Exception:
        return 0


def _orders_date_range() -> tuple:
    if _orders_count() == 0:
        return ("N/A", "N/A")
    try:
        conn = sqlite3.connect(_db_path())
        mn = conn.execute("SELECT MIN(created_at) FROM orders").fetchone()[0]
        mx = conn.execute("SELECT MAX(created_at) FROM orders").fetchone()[0]
        conn.close()
        return (mn[:19] if mn else "N/A", mx[:19] if mx else "N/A")
    except Exception:
        return ("N/A", "N/A")


def _sev_badge(sev: str) -> str:
    color = SEVERITY_COLOR.get(sev, TXT_MUTED)
    emoji = SEVERITY_EMOJI.get(sev, "⚪")
    return f'{emoji} <span style="color:{color};font-weight:600">{sev}</span>'


def get_nodes_status_live():
    """Always reads fresh state — never cached."""
    from db.database_manager import get_all_nodes_status
    import sqlite3
    nodes = get_all_nodes_status()
    # Add live row count for each node
    for node in nodes:
        try:
            if os.path.exists(node['path']):
                conn = sqlite3.connect(node['path'])
                count = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
                conn.close()
                node['row_count'] = count
            else:
                node['row_count'] = 0
        except Exception:
            node['row_count'] = 0
    return nodes


def render_slack_preview(incident: dict) -> str:
    """Generate a self-contained HTML Slack message preview."""
    severity = incident.get('severity', 'NONE')
    severity_colors = {
        'CRITICAL': '#da3633', 'HIGH': '#da3633',
        'MEDIUM': '#d29922', 'LOW': '#238636', 'NONE': '#8b949e'
    }
    color = severity_colors.get(severity, '#8b949e')

    anomaly = incident.get('anomaly_type', 'unknown')
    root_cause = incident.get('root_cause', 'N/A')
    action = incident.get('fix_taken', 'N/A')
    confidence = incident.get('confidence', 0)
    gap = incident.get('gap_minutes', 0)
    run_id = incident.get('run_id', 'N/A')
    timestamp = incident.get('timestamp', 'N/A')
    tables = incident.get('affected_tables', '[]')

    html = f"""
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
                background:#1a1d21;border-radius:8px;padding:0;max-width:680px;
                border-left:4px solid {color};overflow:hidden;">
      <div style="padding:12px 16px;background:#222529;border-bottom:1px solid #383b41;">
        <span style="color:#e8e8e8;font-weight:700;font-size:15px;">
          \U0001f6a8 Pipeline Alert — {severity}
        </span>
      </div>
      <div style="padding:16px;">
        <table style="width:100%;border-collapse:collapse;">
          <tr>
            <td style="color:#ababad;font-size:13px;padding:4px 0;width:140px;">Pipeline</td>
            <td style="color:#e8e8e8;font-size:13px;font-family:monospace;">orders_ingestion</td>
          </tr>
          <tr>
            <td style="color:#ababad;font-size:13px;padding:4px 0;">Detected At</td>
            <td style="color:#e8e8e8;font-size:13px;font-family:monospace;">{timestamp}</td>
          </tr>
          <tr>
            <td style="color:#ababad;font-size:13px;padding:4px 0;">Anomaly Type</td>
            <td style="color:#e8e8e8;font-size:13px;">{anomaly}</td>
          </tr>
          <tr>
            <td style="color:#ababad;font-size:13px;padding:4px 0;">Severity</td>
            <td><span style="background:{color};color:white;padding:2px 8px;
                             border-radius:3px;font-size:12px;font-weight:600;">{severity}</span></td>
          </tr>
          <tr><td colspan="2" style="padding:8px 0;">
            <div style="border-top:1px solid #383b41;"></div>
          </td></tr>
          <tr>
            <td style="color:#ababad;font-size:13px;padding:4px 0;">Root Cause</td>
            <td style="color:#e8e8e8;font-size:13px;">{root_cause}</td>
          </tr>
          <tr>
            <td style="color:#ababad;font-size:13px;padding:4px 0;">Action Taken</td>
            <td style="color:#e8e8e8;font-size:13px;">{action}</td>
          </tr>
          <tr>
            <td style="color:#ababad;font-size:13px;padding:4px 0;">Confidence</td>
            <td style="color:#e8e8e8;font-size:13px;">{int(float(confidence)*100) if confidence else 0}%</td>
          </tr>
          <tr><td colspan="2" style="padding:8px 0;">
            <div style="border-top:1px solid #383b41;"></div>
          </td></tr>
          <tr>
            <td style="color:#ababad;font-size:13px;padding:4px 0;">Affected Tables</td>
            <td style="color:#e8e8e8;font-size:13px;font-family:monospace;">{tables}</td>
          </tr>
          <tr>
            <td style="color:#ababad;font-size:13px;padding:4px 0;">Gap (min)</td>
            <td style="color:#e8e8e8;font-size:13px;">{gap}</td>
          </tr>
        </table>
        <div style="margin-top:12px;padding:8px 12px;background:#2c2f33;
                    border-radius:4px;font-family:monospace;font-size:12px;color:#ababad;">
          Run ID: {run_id}
        </div>
      </div>
    </div>
    """
    return html


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🚨 Pipeline Monitor")
    st.caption("Interactive Control Center")
    st.divider()

    # ── Configuration ──────────────────────────────────────────────────────
    st.subheader("⚙️ Configuration")

    env_path = os.path.join(_PROJECT_ROOT, ".env")

    mock_val = st.toggle("MOCK_MODE", value=MOCK_MODE, help="Skip Groq LLM & Slack webhook calls")

    groq_key = st.text_input(
        "GROQ_API_KEY", type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        placeholder="gsk_...",
    )
    slack_url = st.text_input(
        "SLACK_WEBHOOK_URL", type="password",
        value=os.getenv("SLACK_WEBHOOK_URL", ""),
        placeholder="https://hooks.slack.com/...",
    )

    if st.button("💾 Save Config", use_container_width=True):
        try:
            dotenv_set_key(env_path, "MOCK_MODE", str(mock_val))
            dotenv_set_key(env_path, "GROQ_API_KEY", groq_key)
            dotenv_set_key(env_path, "SLACK_WEBHOOK_URL", slack_url)
            st.success("Config saved to .env — restart app to apply")
        except Exception as exc:
            st.error(f"Save failed: {exc}")

    # Status indicators
    groq_ok = bool(groq_key and groq_key.startswith("gsk_"))
    slack_ok = bool(slack_url and slack_url.startswith("https://"))
    st.markdown(
        f"{'🟢' if groq_ok else '🔴'} Groq {'connected' if groq_ok else 'not set'}  \n"
        f"{'🟢' if slack_ok else '🔴'} Slack {'connected' if slack_ok else 'not set'}"
    )

    st.divider()

    # ── Database ───────────────────────────────────────────────────────────
    st.subheader("🗄️ Database")
    st.markdown(f"**Path:** `{_db_path()}`")
    cnt = _orders_count()
    mn, mx = _orders_date_range()
    st.markdown(f"**Orders:** {cnt:,}  \n**Range:** {mn} → {mx}")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("🌱 Seed DB", use_container_width=True):
            with st.spinner("Seeding 30 days of data…"):
                try:
                    from seed_db import seed_database
                    res = seed_database(_db_path())
                    st.success(f"✅ {res['orders_inserted']:,} orders seeded")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Seed failed: {exc}")

    with col_s2:
        if st.button("🗑️ Clear+Re", use_container_width=True, help="Clear & Reseed"):
            with st.spinner("Clearing & re-seeding…"):
                try:
                    from seed_db import seed_database
                    res = seed_database(_db_path())
                    st.success(f"✅ Re-seeded {res['orders_inserted']:,} orders")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Reseed failed: {exc}")

    st.divider()
    auto_refresh = st.checkbox("🔄 Auto-refresh (60s)", value=st.session_state.get("auto_refresh", False))
    st.session_state["auto_refresh"] = auto_refresh

    st.divider()

    # ── Cluster Control ────────────────────────────────────────────────────
    st.subheader("🗄️ Cluster Control")
    try:
        from db.database_manager import reset_all_nodes
        # Re-read cluster state — this key changes after every pipeline step
        _ = st.session_state.get('last_cluster_check', 0)

        nodes = get_nodes_status_live()
        active = next((n for n in nodes if n['status'] == 'active'), nodes[0])
        failed = [n for n in nodes if n['status'] == 'failed']

        # Active node with color
        st.markdown(
            f"Active: <span style='color:{active['color']};font-weight:bold'>"
            f"{active['label']}</span>",
            unsafe_allow_html=True,
        )

        # Show failed nodes if any
        if failed:
            for f in failed:
                st.markdown(
                    f"<span style='color:#da3633'>✗ {f['label']} — FAILED</span>",
                    unsafe_allow_html=True,
                )

        if st.button("🔄 Reset Cluster → PRIMARY", use_container_width=True):
            reset_all_nodes()
            st.success("Cluster reset to PRIMARY")
            st.rerun()

        st.markdown(
            "[PRIMARY :8080](http://localhost:8080) · "
            "[REPLICA-1 :8081](http://localhost:8081) · "
            "[REPLICA-2 :8082](http://localhost:8082)"
        )
    except Exception:
        st.caption("Cluster module not loaded")


# ═════════════════════════════════════════════════════════════════════════════
# GUARD — DB must exist
# ═════════════════════════════════════════════════════════════════════════════

if not _db_exists():
    st.warning("⚠️ Database not found. Click **🌱 Seed DB** in the sidebar to create it.")
    st.stop()

# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════

tab_overview, tab_sim, tab_intel, tab_slack, tab_cluster = st.tabs([
    "🏠 Overview", "🎬 Run Simulation", "📊 Agent Intelligence", "🔔 Slack Preview", "🗄️ Cluster"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

with tab_overview:
    st.header("🏠 Pipeline Overview")

    # ── KPI row ────────────────────────────────────────────────────────────
    try:
        conn = sqlite3.connect(_db_path())
        cutoff_7d = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        cutoff_30d = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

        total_7d = conn.execute(
            "SELECT COUNT(*) FROM incidents WHERE timestamp > ? AND anomaly_type != ''",
            (cutoff_7d,),
        ).fetchone()[0]

        total_runs_7d = conn.execute(
            "SELECT COUNT(*) FROM incidents WHERE timestamp > ?", (cutoff_7d,),
        ).fetchone()[0]
        det_rate = total_7d / total_runs_7d if total_runs_7d > 0 else 0.0

        mttr_row = conn.execute(
            """SELECT AVG((julianday(resolved_at) - julianday(timestamp))*24*60)
               FROM incidents WHERE resolved=1 AND resolved_at IS NOT NULL AND timestamp > ?""",
            (cutoff_30d,),
        ).fetchone()
        mttr_val = float(mttr_row[0]) if mttr_row and mttr_row[0] else 0.0
        conn.close()
    except Exception:
        total_7d, det_rate, mttr_val = 0, 0.0, 0.0

    k1, k2, k3 = st.columns(3)
    k1.metric("Incidents (7d)", f"{total_7d:,}")
    k2.metric("Detection Rate", f"{det_rate:.0%}")
    k3.metric("Avg MTTR", f"{mttr_val:.1f} min")

    # ── Pipeline status banner ─────────────────────────────────────────────
    try:
        conn = sqlite3.connect(_db_path())
        last = conn.execute(
            "SELECT anomaly_type, severity, timestamp FROM incidents ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        conn.close()
    except Exception:
        last = None

    if last and last[0]:
        sev = last[1] or "NONE"
        ts = last[2][:19] if last[2] else "?"
        st.markdown(
            f'<div class="banner-bad">'
            f'<b style="font-size:1.2rem">{SEVERITY_EMOJI.get(sev,"🔴")} ANOMALY DETECTED</b>'
            f' &nbsp;—&nbsp; <code>{last[0]}</code> | Severity: <b>{sev}</b>'
            f'<br><span class="mono">Last checked: {ts}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        ts = last[2][:19] if last else "never"
        st.markdown(
            f'<div class="banner-ok">'
            f'<b style="font-size:1.2rem">✅ Pipeline Healthy</b>'
            f'<br><span class="mono">Last checked: {ts}</span></div>',
            unsafe_allow_html=True,
        )

    # ── Charts row ─────────────────────────────────────────────────────────
    ch_left, ch_right = st.columns([3, 2])

    with ch_left:
        st.subheader("📈 Incident Timeline (30d)")
        try:
            conn = sqlite3.connect(_db_path())
            rows = conn.execute(
                "SELECT DATE(timestamp) as day, COUNT(*) as cnt FROM incidents "
                "WHERE timestamp > ? GROUP BY DATE(timestamp) ORDER BY day",
                (cutoff_30d,),
            ).fetchall()
            conn.close()
            if rows:
                df_t = pd.DataFrame(rows, columns=["date", "count"])
                fig = go.Figure(go.Scatter(
                    x=df_t["date"], y=df_t["count"], fill="tozeroy",
                    line=dict(color=ACCENT, width=2, shape="spline"),
                    fillcolor="rgba(31,111,235,0.15)",
                ))
                fig.update_layout(**PLOTLY_DARK, height=340)
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(showgrid=True, gridcolor='#30363d')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No incident data yet.")
        except Exception as exc:
            st.error(f"Chart error: {exc}")

    with ch_right:
        st.subheader("🎯 Severity Distribution")
        try:
            conn = sqlite3.connect(_db_path())
            rows = conn.execute(
                "SELECT severity, COUNT(*) FROM incidents WHERE timestamp > ? "
                "AND severity IS NOT NULL AND severity != '' GROUP BY severity ORDER BY COUNT(*) DESC",
                (cutoff_30d,),
            ).fetchall()
            conn.close()
            if rows:
                labels = [r[0] for r in rows]
                values = [r[1] for r in rows]
                colors = [SEVERITY_COLOR.get(l, NONE_GRAY) for l in labels]
                fig = go.Figure(go.Pie(
                    labels=labels, values=values, hole=0.6,
                    marker=dict(colors=colors),
                    textinfo="label+percent", textposition="outside",
                ))
                fig.update_layout(**PLOTLY_DARK, height=340, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No severity data yet.")
        except Exception as exc:
            st.error(f"Chart error: {exc}")

    # ── Recent incidents table ─────────────────────────────────────────────
    st.subheader("📋 Recent Incidents")
    try:
        conn = sqlite3.connect(_db_path())
        rows = conn.execute(
            "SELECT run_id, timestamp, anomaly_type, severity, gap_minutes, "
            "root_cause, fix_taken, resolved, confidence "
            "FROM incidents ORDER BY timestamp DESC LIMIT 10"
        ).fetchall()
        conn.close()
        if rows:
            cols = ["Run ID", "Timestamp", "Anomaly", "Severity", "Gap(min)",
                    "Root Cause", "Fix", "Resolved", "Confidence"]
            df_i = pd.DataFrame(rows, columns=cols)
            df_i["Resolved"] = df_i["Resolved"].apply(lambda v: "✅" if v else "❌")
            df_i["Confidence"] = df_i["Confidence"].apply(
                lambda v: f"{v:.0%}" if isinstance(v, (int, float)) and v else "—"
            )
            df_i["Root Cause"] = df_i["Root Cause"].apply(
                lambda v: (v[:60] + "…") if isinstance(v, str) and len(v) > 60 else (v or "—")
            )
            st.dataframe(df_i, use_container_width=True, height=380)
        else:
            st.info("No incidents recorded yet. Run the pipeline to generate data.")
    except Exception as exc:
        st.error(f"Table error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — RUN SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

with tab_sim:
    st.header("🎬 Run Simulation")

    # ── Sub-section A: Severity Scenarios ───────────────────────────────────
    st.subheader("🎯 Test Scenarios")

    sc1, sc2, sc3 = st.columns(3)

    with sc1:
        st.markdown("""
        <div style='border:1px solid #d29922; border-radius:8px; padding:16px; background:#1c1a10'>
            <div style='font-size:24px'>🟡 LOW</div>
            <div style='font-weight:bold; margin:8px 0'>Brief Network Hiccup</div>
            <div style='color:#8b949e; font-size:13px'>15-min data gap. Repairer waits and retries. No failover. PRIMARY stays active.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("▶ Run LOW scenario", key="scenario_low", use_container_width=True):
            st.session_state["pending_scenario"] = "low"

    with sc2:
        st.markdown("""
        <div style='border:1px solid #1f6feb; border-radius:8px; padding:16px; background:#0d1b2e'>
            <div style='font-size:24px'>🟠 MEDIUM</div>
            <div style='font-weight:bold; margin:8px 0'>Mumbai DB Lost (2hrs)</div>
            <div style='color:#8b949e; font-size:13px'>2-hr gap + null spike. Repairer switches pipeline to REPLICA-1. First failover.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("▶ Run MEDIUM scenario", key="scenario_medium", use_container_width=True):
            st.session_state["pending_scenario"] = "medium"

    with sc3:
        st.markdown("""
        <div style='border:1px solid #da3633; border-radius:8px; padding:16px; background:#1e0c0c'>
            <div style='font-size:24px'>🔴 HIGH/CRITICAL</div>
            <div style='font-weight:bold; margin:8px 0'>PRIMARY Corruption</div>
            <div style='color:#8b949e; font-size:13px'>4-hr gap + null storm + schema drift. Full failover to REPLICA-2. Slack fires.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("▶ Run HIGH scenario", key="scenario_high", use_container_width=True):
            st.session_state["pending_scenario"] = "high"

    # Handle pending scenario
    pending = st.session_state.pop("pending_scenario", None)
    if pending:
        try:
            from dashboard.scenarios import run_scenario_low, run_scenario_medium, run_scenario_high
            if pending == "low":
                sc_result = run_scenario_low()
            elif pending == "medium":
                sc_result = run_scenario_medium()
            else:
                sc_result = run_scenario_high()
            st.success(
                f"{sc_result['icon']} **{sc_result['title']}** — "
                f"{sc_result['description']}  \n"
                f"Rows affected: **{sc_result.get('rows_deleted', 0):,}**"
            )
        except Exception as exc:
            st.error(f"Scenario injection error: {traceback.format_exc()}")

    st.divider()

    # ── Manual Injection (advanced) ─────────────────────────────────────────
    with st.expander("💉 Manual Anomaly Injection (Advanced)", expanded=False):
        anomaly_choice = st.radio(
            "Choose anomaly type:",
            ["Healthy Check (no anomaly)", "Missing Data (delete recent rows)",
             "Null Spike (quality issue)", "Schema Drift (add column)"],
            horizontal=True,
        )

        inject_params = {}
        if "Missing Data" in anomaly_choice:
            inject_params["gap_minutes"] = st.slider("Gap size (minutes)", 10, 360, 30)
        elif "Null Spike" in anomaly_choice:
            inject_params["null_pct"] = st.slider("Null percentage", 10, 100, 100)

        inject_btn = st.button(
            "💉 Inject Anomaly",
            use_container_width=True,
            disabled=("Healthy" in anomaly_choice),
        )

        if inject_btn:
            try:
                from dashboard.injectors import (
                    inject_missing_data, inject_null_spike, inject_schema_drift,
                )
                if "Missing Data" in anomaly_choice:
                    res = inject_missing_data(_db_path(), inject_params["gap_minutes"])
                    if res["success"]:
                        st.success(f"✅ Deleted **{res['rows_deleted']:,}** rows (last {res['gap_minutes']} min)")
                    else:
                        st.error(res.get("error", "Injection failed"))
                elif "Null Spike" in anomaly_choice:
                    res = inject_null_spike(_db_path(), inject_params["null_pct"])
                    if res["success"]:
                        st.success(f"✅ Set **{res['rows_affected']:,}** of {res['total_recent']} recent rows to NULL")
                    else:
                        st.error(res.get("error", "Injection failed"))
                elif "Schema Drift" in anomaly_choice:
                    res = inject_schema_drift(_db_path())
                    if res["success"]:
                        st.success(f"✅ Added column `{res['column_name']}` to orders table")
                    else:
                        st.error(res.get("error", "Injection failed"))
            except Exception as exc:
                st.error(f"Injection error: {traceback.format_exc()}")

    st.divider()

    # ── Sub-section B: Run Pipeline ────────────────────────────────────────
    st.subheader("▶️ Run Pipeline")

    can_run = _orders_count() > 0
    if not can_run:
        st.warning("⚠️ No data in orders table. Seed the database first.")

    run_btn = st.button(
        "▶️ Run Pipeline Now", type="primary",
        use_container_width=True, disabled=not can_run,
    )

    if run_btn:
        import config as _cfg
        _cfg.MOCK_MODE = mock_val
        _cfg.GROQ_API_KEY = groq_key or _cfg.GROQ_API_KEY
        _cfg.SLACK_WEBHOOK_URL = slack_url or _cfg.SLACK_WEBHOOK_URL

        from agents.monitor import MonitorAgent
        from agents.diagnoser import DiagnoserAgent
        from agents.repairer import RepairerAgent
        from agents.slack_agent import SlackAgent
        from state import create_initial_state
        from memory.incident_store import insert_incident

        run_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now(timezone.utc).isoformat()
        state = create_initial_state(run_id=run_id, timestamp=timestamp)

        with st.status("🔄 Pipeline running…", expanded=True) as status:
            # ── Step 1: Monitor ────────────────────────────────────────────
            st.write("🔍 **Step 1: Monitor Agent** — checking database…")
            try:
                monitor = MonitorAgent(db_path=_db_path())
                state = monitor.run(state)
                st.session_state['last_cluster_check'] = time.time()
                st.write(
                    f"&nbsp;&nbsp;&nbsp;→ Count: **{state['raw_count']}** rows | "
                    f"Baseline: **{state['expected_avg']:.0f}** | "
                    f"Null rate: **{state['null_rate']:.2%}**"
                )
                if state["anomaly_detected"]:
                    sev = state["severity"]
                    st.write(
                        f"&nbsp;&nbsp;&nbsp;→ {SEVERITY_EMOJI.get(sev,'⚠️')} "
                        f"**ANOMALY:** `{state['anomaly_type']}` | "
                        f"Severity: **{sev}**"
                    )
                else:
                    st.write("&nbsp;&nbsp;&nbsp;→ ✅ No anomaly detected")
            except Exception as exc:
                st.error(f"Monitor failed: {exc}")
                status.update(label="❌ Pipeline failed at Monitor", state="error")
                st.session_state["last_run_state"] = state
                st.stop()

            # ── Steps 2-4 only if anomaly ──────────────────────────────────
            if state["anomaly_detected"]:
                # Step 2: Diagnoser
                st.write("🧠 **Step 2: Diagnoser Agent** — analyzing root cause…")
                try:
                    diagnoser = DiagnoserAgent()
                    state = diagnoser.run(state)
                    st.session_state['last_cluster_check'] = time.time()
                    diag = state.get("diagnoser_output", {})
                    st.write(
                        f"&nbsp;&nbsp;&nbsp;→ Root cause: *{diag.get('root_cause','?')}*"
                    )
                    st.write(
                        f"&nbsp;&nbsp;&nbsp;→ Confidence: **{diag.get('confidence',0)*100:.0f}%**"
                    )
                except Exception as exc:
                    st.error(f"Diagnoser failed: {exc}")

                # Step 3: Repairer
                st.write("🔧 **Step 3: Repairer Agent** — applying fix…")
                try:
                    repairer = RepairerAgent(db_path=_db_path())
                    state = repairer.run(state)
                    st.session_state['last_cluster_check'] = time.time()
                    rep = state.get("repairer_output", {})
                    ok = "✅" if rep.get("success") else "❌"
                    st.write(
                        f"&nbsp;&nbsp;&nbsp;→ Action: `{rep.get('action_taken','?')}` | "
                        f"Success: {ok} | Rows: {rep.get('rows_affected',0)}"
                    )
                except Exception as exc:
                    st.error(f"Repairer failed: {exc}")

                # ── Failover display ───────────────────────────────────────
                if state.get("failover_result") and state["failover_result"].get("success"):
                    fr = state["failover_result"]
                    st.markdown(f"""
                    <div style='border-left:3px solid #da3633; padding:12px; background:#1e0c0c; margin:8px 0; border-radius:4px'>
                        ⚡ <b>FAILOVER EXECUTED</b><br>
                        <span style='color:#8b949e'>{fr['from_node']['label']}</span>
                        → <span style='color:#238636; font-weight:bold'>{fr['to_node']['label']}</span><br>
                        <span style='color:#8b949e; font-size:12px'>Data synced to replica. Pipeline reconnected.</span>
                    </div>
                    """, unsafe_allow_html=True)
                elif state.get("failover_result") and state["failover_result"].get("error") == "ALL_NODES_FAILED":
                    st.error("🚨 **ALL NODES FAILED** — No healthy replicas. Manual intervention required.")

                # Step 4: Slack
                st.write("📨 **Step 4: Slack Agent** — sending report…")
                try:
                    slack = SlackAgent(webhook_url=slack_url if not mock_val else "")
                    state = slack.run(state)
                    st.session_state['last_cluster_check'] = time.time()
                    if state.get("slack_sent"):
                        st.write("&nbsp;&nbsp;&nbsp;→ ✅ Slack message sent")
                    else:
                        st.write("&nbsp;&nbsp;&nbsp;→ ⚠️ Slack skipped (MOCK_MODE or no webhook)")
                except Exception as exc:
                    st.error(f"Slack agent failed: {exc}")

            status.update(
                label="✅ Pipeline run complete" if not state["anomaly_detected"]
                else f"✅ Pipeline complete — {state['anomaly_type']} ({state['severity']})",
                state="complete",
            )

        # ── Persist incident (update with final fix_taken / resolved) ──────
        try:
            diag = state.get("diagnoser_output", {})
            insert_incident({
                "run_id": run_id,
                "timestamp": timestamp,
                "anomaly_type": state.get("anomaly_type", ""),
                "severity": state.get("severity", "NONE"),
                "gap_minutes": state.get("gap_minutes", 0.0),
                "root_cause": diag.get("root_cause", ""),
                "affected_tables": state.get("affected_tables", []),
                "fix_taken": state.get("repairer_output", {}).get("action_taken", ""),
                "resolved": int(state.get("repairer_output", {}).get("success", False)),
                "confidence": diag.get("confidence", 0.0),
            }, db_path=_db_path())
        except Exception:
            pass

        # ── Save for Slack Preview tab ─────────────────────────────────────
        st.session_state["last_run_state"] = dict(state)

        # ── Run report ─────────────────────────────────────────────────────
        with st.expander("📄 Full Run Report", expanded=False):
            report_cols = st.columns(2)
            with report_cols[0]:
                st.markdown(f"**Run ID:** `{run_id}`")
                st.markdown(f"**Timestamp:** `{timestamp}`")
                st.markdown(f"**Anomaly:** `{state.get('anomaly_type') or 'none'}`")
                st.markdown(f"**Severity:** {_sev_badge(state.get('severity','NONE'))}", unsafe_allow_html=True)
                st.markdown(f"**Raw Count:** {state.get('raw_count',0)}")
                st.markdown(f"**Baseline:** {state.get('expected_avg',0):.1f}")
                st.markdown(f"**Null Rate:** {state.get('null_rate',0):.2%}")
            with report_cols[1]:
                st.markdown("**Diagnoser Output:**")
                st.json(state.get("diagnoser_output", {}))
                st.markdown("**Repairer Output:**")
                st.json(state.get("repairer_output", {}))
                st.markdown(f"**Slack Sent:** {'✅' if state.get('slack_sent') else '❌'}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — AGENT INTELLIGENCE
# ─────────────────────────────────────────────────────────────────────────────

with tab_intel:
    st.header("📊 Agent Intelligence")

    # ── Episodic Memory ────────────────────────────────────────────────────
    st.subheader("🧠 Episodic Memory (Incident History)")

    # Filters
    f1, f2, f3 = st.columns(3)
    with f1:
        filt_type = st.selectbox("Anomaly Type", ["All", "missing_data", "data_quality", "schema_drift"])
    with f2:
        filt_sev = st.selectbox("Severity", ["All", "LOW", "MEDIUM", "HIGH", "CRITICAL"])
    with f3:
        filt_days = st.number_input("Last N days", min_value=1, max_value=365, value=30)

    try:
        conn = sqlite3.connect(_db_path())
        cutoff = (datetime.now(timezone.utc) - timedelta(days=filt_days)).isoformat()
        q = "SELECT * FROM incidents WHERE timestamp > ?"
        params = [cutoff]
        if filt_type != "All":
            q += " AND anomaly_type = ?"
            params.append(filt_type)
        if filt_sev != "All":
            q += " AND severity = ?"
            params.append(filt_sev)
        q += " ORDER BY timestamp DESC"
        df_inc = pd.read_sql_query(q, conn, params=params)
        conn.close()

        if not df_inc.empty:
            st.dataframe(df_inc, use_container_width=True, height=350)

            # Resolve button
            unresolved = df_inc[
                (df_inc["resolved"] == 0)
                & (df_inc["severity"].isin(["HIGH", "CRITICAL"]))
            ]
            if not unresolved.empty:
                resolve_id = st.selectbox(
                    "Mark as Resolved:",
                    unresolved["run_id"].tolist(),
                    key="resolve_select",
                )
                if st.button("✅ Resolve Incident", key="resolve_btn"):
                    try:
                        from memory.incident_store import auto_resolve_incident
                        ok = auto_resolve_incident(resolve_id, db_path=_db_path())
                        if ok:
                            st.success(f"Incident `{resolve_id}` resolved")
                            st.rerun()
                        else:
                            st.warning("Incident not found or already resolved")
                    except Exception as exc:
                        st.error(f"Failed: {exc}")
        else:
            st.info("No incidents match the filter.")
    except Exception as exc:
        st.error(f"Error: {exc}")

    st.divider()

    # ── Procedural Memory (Playbooks) ──────────────────────────────────────
    st.subheader("📚 Procedural Memory (Playbooks)")
    try:
        conn = sqlite3.connect(_db_path())
        rows = conn.execute(
            "SELECT anomaly_type, severity, action_taken, success_count, failure_count, last_used "
            "FROM playbooks ORDER BY CAST(success_count AS REAL)/MAX(success_count+failure_count,1) DESC"
        ).fetchall()
        conn.close()

        if rows:
            pb_data = []
            for r in rows:
                total = r[3] + r[4]
                rate = r[3] / total if total > 0 else 0
                pb_data.append({
                    "Anomaly": r[0],
                    "Severity": r[1],
                    "Action": r[2],
                    "Wins": r[3],
                    "Losses": r[4],
                    "Success %": f"{rate:.0%}",
                    "Last Used": (r[5] or "—")[:19],
                })
            df_pb = pd.DataFrame(pb_data)
            st.dataframe(df_pb, use_container_width=True)

            # Mini bar chart
            for d in pb_data:
                rate_val = int(d["Success %"].replace("%", ""))
                color = SUCCESS if rate_val > 70 else (WARNING if rate_val >= 40 else DANGER)
                st.markdown(
                    f'`{d["Action"]}` ({d["Anomaly"]}/{d["Severity"]}): '
                    f'<div style="background:{BORDER};border-radius:4px;height:14px;width:100%;display:inline-block;">'
                    f'<div style="background:{color};height:100%;width:{rate_val}%;border-radius:4px;"></div></div>'
                    f' **{d["Success %"]}**',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No playbook entries yet. Run the pipeline to build procedural memory.")
    except Exception as exc:
        st.error(f"Playbook error: {exc}")

    st.divider()

    # ── Schema Registry ────────────────────────────────────────────────────
    st.subheader("📐 Schema Registry")
    try:
        from memory.schema_registry import load_expected_schema, get_actual_schema, check_drift

        expected = load_expected_schema("orders")
        conn = sqlite3.connect(_db_path())
        actual = get_actual_schema(conn, "orders")
        drift = check_drift(conn, "orders")
        conn.close()

        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("**Expected Schema** (`schemas/orders.json`)")
            if expected and "columns" in expected:
                exp_df = pd.DataFrame(expected["columns"])
                st.dataframe(exp_df, use_container_width=True)
            else:
                st.warning("No expected schema found.")
        with sc2:
            st.markdown("**Actual Schema** (`PRAGMA table_info`)")
            if actual:
                act_df = pd.DataFrame(actual)
                st.dataframe(act_df, use_container_width=True)
            else:
                st.warning("Could not read actual schema.")

        if drift["drift_detected"]:
            st.error("🚨 **DRIFT DETECTED**")
            if drift["new_columns"]:
                st.markdown(f"**New columns:** {', '.join(f'`{c}`' for c in drift['new_columns'])}")
            if drift["deleted_columns"]:
                st.markdown(f"**Deleted columns:** {', '.join(f'`{c}`' for c in drift['deleted_columns'])}")
            if drift["type_changes"]:
                for tc in drift["type_changes"]:
                    st.markdown(f"**Type change:** `{tc['column']}` {tc['expected']} → {tc['actual']}")
            if drift["nullable_changes"]:
                for nc in drift["nullable_changes"]:
                    st.markdown(f"**Nullable change:** `{nc['column']}`")

            if st.button("🔄 Reset Schema to Current"):
                try:
                    schema_path = os.path.join(_PROJECT_ROOT, "schemas", "orders.json")
                    new_schema = {
                        "table": "orders",
                        "version": "1.1",
                        "columns": [
                            {
                                "name": c["name"],
                                "type": c["type"],
                                "nullable": not c["notnull"],
                                "primary_key": False,
                            }
                            for c in actual
                        ],
                    }
                    with open(schema_path, "w") as f:
                        json.dump(new_schema, f, indent=4)
                    st.success("Schema baseline updated to match current DB")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed: {exc}")
        else:
            st.success("✅ No schema drift detected — expected and actual schemas match.")
    except Exception as exc:
        st.error(f"Schema registry error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — SLACK PREVIEW
# ─────────────────────────────────────────────────────────────────────────────

with tab_slack:
    st.header("🔔 Slack Preview")

    last_state = st.session_state.get("last_run_state")

    if not last_state or not last_state.get("anomaly_detected"):
        # Try loading most recent anomaly from DB
        try:
            conn = sqlite3.connect(_db_path())
            row = conn.execute(
                "SELECT run_id, timestamp, anomaly_type, severity, gap_minutes, "
                "root_cause, affected_tables, fix_taken, resolved, confidence "
                "FROM incidents WHERE anomaly_type != '' ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            conn.close()
            if row:
                last_state = {
                    "run_id": row[0], "timestamp": row[1],
                    "anomaly_type": row[2], "severity": row[3],
                    "anomaly_detected": True,
                    "gap_minutes": row[4] or 0,
                    "affected_tables": json.loads(row[6]) if row[6] else [],
                    "diagnoser_output": {"root_cause": row[5] or "Unknown", "confidence": row[9] or 0},
                    "repairer_output": {"action_taken": row[7] or "none", "success": bool(row[8])},
                    "slack_sent": False,
                }
        except Exception:
            pass

    if not last_state or not last_state.get("anomaly_detected"):
        st.info(
            "No anomaly data available. Run a simulation with an injected anomaly "
            "to preview the Slack message."
        )
    else:
        sev = last_state.get("severity", "NONE")
        diag = last_state.get("diagnoser_output", {})
        rep = last_state.get("repairer_output", {})
        atype = last_state.get("anomaly_type", "unknown")
        run_id = last_state.get("run_id", "?")
        ts_raw = last_state.get("timestamp", "")

        # Format timestamp
        try:
            from prompts.slack_template import _timestamp_ist
            ts_display = _timestamp_ist(ts_raw)
        except Exception:
            ts_display = ts_raw[:19] if ts_raw else "N/A"

        tables = ", ".join(last_state.get("affected_tables", [])) or "N/A"
        conf = diag.get("confidence", 0)

        # ── Slack-style HTML preview using st.components.v1.html ────────────
        st.markdown("### Message Preview")
        is_critical = sev == "CRITICAL"

        import streamlit.components.v1 as components
        latest_incident = {
            'severity': sev,
            'anomaly_type': atype,
            'root_cause': diag.get('root_cause', 'Unknown'),
            'fix_taken': rep.get('action_taken', 'none'),
            'confidence': conf,
            'gap_minutes': last_state.get('gap_minutes', 0),
            'run_id': run_id,
            'timestamp': ts_display,
            'affected_tables': tables,
        }
        components.html(render_slack_preview(latest_incident), height=420, scrolling=False)

        st.markdown("")

        # ── Send button ────────────────────────────────────────────────────
        s1, s2 = st.columns([1, 3])
        with s1:
            send_btn = st.button(
                "📤 Send Now", type="primary",
                use_container_width=True,
                disabled=not slack_ok,
            )
        with s2:
            if not slack_ok:
                st.caption("⚠️ Set SLACK_WEBHOOK_URL in the sidebar to enable sending.")

        if send_btn:
            try:
                from agents.slack_agent import force_send_slack
                ok = force_send_slack(last_state, slack_url)
                if ok:
                    st.success(f"✅ Slack message sent at {datetime.now().strftime('%H:%M:%S')}")
                else:
                    st.error("❌ Slack send failed — check webhook URL")
            except Exception as exc:
                st.error(f"Send error: {exc}")

        # ── Raw payload ────────────────────────────────────────────────────
        with st.expander("📋 Raw Block Kit JSON"):
            try:
                from prompts.slack_template import format_slack_message, format_escalation_message
                if is_critical:
                    payload = format_escalation_message(last_state)
                else:
                    payload = format_slack_message(last_state)
                st.json(payload)
            except Exception as exc:
                st.error(str(exc))

        # ── Setup guide if no webhook ──────────────────────────────────────
        if not slack_ok:
            st.divider()
            st.subheader("📖 Slack Webhook Setup Guide")
            st.markdown("""
1. Go to [api.slack.com/apps](https://api.slack.com/apps) and click **Create New App**
2. Choose **From scratch**, name it "Pipeline Monitor", select your workspace
3. Under **Features → Incoming Webhooks**, toggle it **On**
4. Click **Add New Webhook to Workspace**, select a channel
5. Copy the webhook URL (starts with `https://hooks.slack.com/services/...`)
6. Paste it in the **SLACK_WEBHOOK_URL** field in the sidebar
7. Click **💾 Save Config**
            """)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — CLUSTER
# ─────────────────────────────────────────────────────────────────────────────

with tab_cluster:
    st.header("🗄️ Cluster Topology")

    try:
        from db.database_manager import (
            get_failover_state, get_failover_log,
            _load_state,
        )

        # Force re-read on every render — no @st.cache_data here
        nodes = get_nodes_status_live()
        fo_state = _load_state()

        # ── Section A: Live Topology Visual ────────────────────────────────
        def _render_node_card(node: dict) -> str:
            """Generate HTML for a single node card."""
            status = node["status"]
            color = node["color"]
            label = node["label"]
            port = node["port"]
            row_count = node.get('row_count', 0)

            if status == "active":
                border = f"2px solid {color}"
                badge_bg = f"{color}33"
                badge_text = "🟢 ACTIVE"
                glow = f"0 0 12px {color}55"
                opacity = "1"
                overlay = ""
            elif status == "failed":
                border = f"2px solid {DANGER}"
                badge_bg = f"{DANGER}33"
                badge_text = "🔴 FAILED"
                glow = "none"
                opacity = "0.5"
                overlay = (
                    "<div style='position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);"
                    "font-size:48px;opacity:0.3'>✗</div>"
                )
            else:
                border = f"1px solid {BORDER}"
                badge_bg = f"{BORDER}33"
                badge_text = "🔵 STANDBY"
                glow = "none"
                opacity = "0.6"
                overlay = ""

            return f"""
            <div style='
                position:relative;
                background:{CARD_BG};
                border:{border};
                border-radius:12px;
                padding:16px;
                text-align:center;
                opacity:{opacity};
                box-shadow:{glow};
                {"animation:pulse 2s infinite;" if status == "active" else ""}
            '>
                {overlay}
                <div style='font-weight:700;font-size:1.1rem;color:{TXT};margin-bottom:6px'>{label}</div>
                <div style='
                    display:inline-block;
                    background:{badge_bg};
                    border-radius:12px;
                    padding:2px 10px;
                    font-size:.75rem;
                    color:{TXT};
                    margin-bottom:8px;
                '>{badge_text}</div>
                <div style='color:{TXT_MUTED};font-size:.8rem;font-family:monospace'>
                    :{port}<br>
                    {row_count:,} rows
                </div>
            </div>
            """

        cards_html = "".join(_render_node_card(n) for n in nodes)
        active_label = fo_state.get("active_node_id", "primary").upper()

        topology_html = f"""
        <style>
            @keyframes pulse {{
                0%, 100% {{ box-shadow: 0 0 0 0 rgba(35,134,54,0.4); }}
                70% {{ box-shadow: 0 0 0 10px rgba(35,134,54,0); }}
            }}
        </style>
        <div style='background:{BG};border:1px solid {BORDER};border-radius:12px;padding:20px'>
            <div style='text-align:center;color:{TXT_MUTED};font-size:.85rem;margin-bottom:8px'>
                Pipeline Monitor → DB Cluster
            </div>
            <div style='text-align:center;margin-bottom:16px'>
                <span style='color:{TXT};font-size:.9rem'>Traffic →</span>
                <span style='color:{SUCCESS};font-weight:700;font-size:1rem'> {active_label}</span>
            </div>
            <div style='
                display:grid;
                grid-template-columns:repeat(3,1fr);
                gap:16px;
            '>
                {cards_html}
            </div>
            <div style='text-align:center;color:{TXT_MUTED};font-size:.75rem;margin-top:12px'>
                Failover count: {fo_state.get("failover_count", 0)} |
                Last failover: {(fo_state.get("last_failover") or "never")[:19]}
            </div>
        </div>
        """

        import streamlit.components.v1 as components
        components.html(topology_html, height=280)

        st.markdown("")

        # ── Section B: Failover Event Log ──────────────────────────────────
        st.subheader("📜 Failover Event Log")
        fo_log = get_failover_log()

        if fo_log:
            log_data = []
            for i, entry in enumerate(reversed(fo_log)):
                sev = entry.get("severity", "NONE")
                log_data.append({
                    "Time": (entry.get("timestamp") or "?")[:19],
                    "From": entry.get("from_node", "?"),
                    "→": "→",
                    "To": entry.get("to_node", "?"),
                    "Reason": (entry.get("reason") or "—")[:50],
                    "Severity": sev,
                    "#": entry.get("failover_count", "?"),
                })

            df_log = pd.DataFrame(log_data)
            st.dataframe(df_log, use_container_width=True, height=280)
        else:
            st.info(
                "No failover events yet. Run a **MEDIUM** or **HIGH** scenario "
                "from the Run Simulation tab to trigger a failover."
            )

    except Exception as exc:
        st.error(f"Cluster tab error: {exc}")
        st.caption("Make sure the `db/database_manager.py` module exists.")

    # Auto-refresh the cluster tab every 3 seconds when a simulation is running
    if st.session_state.get('pipeline_running', False):
        time.sleep(3)
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# AUTO-REFRESH
# ═════════════════════════════════════════════════════════════════════════════

if st.session_state.get("auto_refresh", False):
    time.sleep(60)
    st.rerun()
