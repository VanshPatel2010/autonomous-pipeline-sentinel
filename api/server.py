import json
import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any
import asyncio
import threading
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
# Make sure we can import from the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.client import get_db_connection
from main import run_pipeline

app = FastAPI(title="Autonomous Pipeline API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Webhook state
webhook_last_received = datetime.now(timezone.utc)
DEAD_MAN_SWITCH_SECONDS = 60

async def dead_man_switch():
    global webhook_last_received
    while True:
        await asyncio.sleep(5)
        if webhook_last_received:
            elapsed = (datetime.now(timezone.utc) - webhook_last_received).total_seconds()
            if elapsed > DEAD_MAN_SWITCH_SECONDS:
                print(f"DEAD MAN SWITCH TRIGGERED! No webhooks for {elapsed:.1f}s")
                # Trigger the agent for a volume outage
                run_pipeline({
                    "anomaly_detected": True,
                    "anomaly_type": "volume_outage",
                    "severity": "HIGH",
                    "gap_minutes": elapsed / 60.0,
                    "affected_tables": ["orders"]
                })
                # Reset to prevent endless triggering until the next webhook
                webhook_last_received = None

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(dead_man_switch())

class InjectRequest(BaseModel):
    inject_gap: bool
    inject_nulls: bool

class WebhookPayload(BaseModel):
    type: str
    table: str
    record: dict
    old_record: dict = None

@app.get("/api/metrics")
def get_metrics():
    """Get the live data stream grouped by minute for the chart."""
    conn = get_db_connection()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    window_start = now - timedelta(minutes=60)
    
    # Get total orders in the last hour
    orders_query = "SELECT created_at, order_amount FROM orders WHERE created_at > ?"
    orders = conn.execute(orders_query, (window_start.isoformat(),)).fetchall()
    
    # Get total quarantined orders in the last hour
    quarantine_query = "SELECT created_at FROM quarantine_orders WHERE created_at > ?"
    quarantined = conn.execute(quarantine_query, (window_start.isoformat(),)).fetchall()
    
    conn.close()

    # Aggregate by minute for the chart
    minute_buckets: Dict[str, Dict[str, int]] = {}
    for i in range(60):
        minute_time = (now - timedelta(minutes=59 - i)).replace(second=0, microsecond=0)
        minute_buckets[minute_time.isoformat()] = {"time": minute_time.strftime("%H:%M"), "orders": 0, "quarantined": 0, "nulls": 0}

    for order in orders:
        # PostgreSQL returns datetime objects, SQLite returns strings
        dt_val = order["created_at"]
        if isinstance(dt_val, str):
            dt = datetime.fromisoformat(dt_val).replace(second=0, microsecond=0)
        else:
            dt = dt_val.replace(second=0, microsecond=0)
            
        bucket_key = dt.isoformat()
        if bucket_key in minute_buckets:
            minute_buckets[bucket_key]["orders"] += 1
            if order["order_amount"] is None:
                minute_buckets[bucket_key]["nulls"] += 1

    for q in quarantined:
        dt_val = q["created_at"]
        if isinstance(dt_val, str):
            dt = datetime.fromisoformat(dt_val).replace(second=0, microsecond=0)
        else:
            dt = dt_val.replace(second=0, microsecond=0)
            
        bucket_key = dt.isoformat()
        if bucket_key in minute_buckets:
            minute_buckets[bucket_key]["quarantined"] += 1

    chart_data = list(minute_buckets.values())
    
    return {
        "chart_data": chart_data[-15:], # Send last 15 mins for the live chart
        "total_orders_1h": len(orders),
        "total_quarantined_1h": len(quarantined)
    }

@app.get("/api/incidents")
def get_incidents():
    """Get the recent incident log from the agent's episodic memory."""
    try:
        conn = get_db_connection()
        incidents = conn.execute("SELECT * FROM incidents ORDER BY timestamp DESC LIMIT 10").fetchall()
        result = [dict(i) for i in incidents]
    except Exception as e:
        print("API Error:", e)
        result = []
    finally:
        if 'conn' in locals():
            conn.close()
    return {"incidents": result}

@app.post("/api/webhook/orders")
async def receive_webhook(payload: WebhookPayload):
    """Receive row-level changes from Supabase."""
    global webhook_last_received
    webhook_last_received = datetime.now(timezone.utc)
    
    record = payload.record
    # Naive check to trigger dbt if something looks suspicious
    if record.get("order_amount") is None:
        print("Webhook detected NULL order_amount! Triggering dbt test...")
        threading.Thread(target=run_pipeline, args=({"anomaly_type": "null_value"},), daemon=True).start()
        
    return {"status": "ok", "received_at": webhook_last_received.isoformat()}

@app.get("/api/simulator/state")
def get_simulator_state():
    try:
        conn = get_db_connection()
        row = conn.execute("SELECT inject_gap, inject_nulls, inject_duplicate FROM simulator_config WHERE id = 1").fetchone()
        conn.close()
        if row:
            return {
                "inject_gap": bool(row["inject_gap"]),
                "inject_nulls": bool(row["inject_nulls"]),
                "inject_duplicate": bool(row.get("inject_duplicate", False))
            }
    except Exception as e:
        print("State fetch error:", e)
    return {"inject_gap": False, "inject_nulls": False, "inject_duplicate": False}

@app.post("/api/simulator/inject")
def inject_anomaly(req: InjectRequest):
    try:
        conn = get_db_connection()
        # Ensure we have all fields including inject_duplicate if available in req, otherwise False
        inject_dup = getattr(req, 'inject_duplicate', False)
        
        conn.execute(
            """
            UPDATE simulator_config 
            SET inject_gap = %s, inject_nulls = %s, inject_duplicate = %s
            WHERE id = 1
            """,
            (req.inject_gap, req.inject_nulls, inject_dup)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print("State update error:", e)
        
    return {"inject_gap": req.inject_gap, "inject_nulls": req.inject_nulls, "inject_duplicate": getattr(req, 'inject_duplicate', False)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
