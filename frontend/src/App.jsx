import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { Activity, AlertTriangle, Zap, CheckCircle, Database } from 'lucide-react';

export default function App() {
  const [metrics, setMetrics] = useState({ chart_data: [], total_orders_1h: 0, total_quarantined_1h: 0 });
  const [incidents, setIncidents] = useState([]);
  const [simState, setSimState] = useState({ inject_gap: false, inject_nulls: false, inject_duplicate: false });

  // Fetch data every 2 seconds
  useEffect(() => {
    const fetchData = async () => {
      try {
        const metRes = await fetch('http://localhost:8000/api/metrics');
        if (metRes.ok) setMetrics(await metRes.json());
        
        const incRes = await fetch('http://localhost:8000/api/incidents');
        if (incRes.ok) {
          const data = await incRes.json();
          setIncidents(data.incidents);
        }

        const simRes = await fetch('http://localhost:8000/api/simulator/state');
        if (simRes.ok) setSimState(await simRes.json());
      } catch (err) {
        console.error("API Error", err);
      }
    };
    fetchData();
    const interval = setInterval(fetchData, 2000);
    return () => clearInterval(interval);
  }, []);

  const toggleAnomaly = async (type) => {
    const newState = { ...simState };
    if (type === 'gap') {
      newState.inject_gap = !newState.inject_gap;
      if (newState.inject_gap) { newState.inject_nulls = false; newState.inject_duplicate = false; }
    }
    if (type === 'nulls') {
      newState.inject_nulls = !newState.inject_nulls;
      if (newState.inject_nulls) { newState.inject_gap = false; newState.inject_duplicate = false; }
    }
    if (type === 'duplicate') {
      newState.inject_duplicate = !newState.inject_duplicate;
      if (newState.inject_duplicate) { newState.inject_gap = false; newState.inject_nulls = false; }
    }
    
    setSimState(newState);
    await fetch('http://localhost:8000/api/simulator/inject', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(newState)
    });
  };

  // Derive some metrics
  const latestData = metrics.chart_data.length > 0 ? metrics.chart_data[metrics.chart_data.length - 1] : { orders: 0, nulls: 0 };
  const currentRate = latestData.orders;
  const nullRate = latestData.orders > 0 ? ((latestData.nulls / latestData.orders) * 100).toFixed(1) : 0;

  return (
    <div className="dashboard-container">
      <div className="header glass-panel" style={{ padding: '1rem 1.5rem', marginBottom: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <Database size={24} color="var(--accent-color)" />
          <h1>Autonomous Pipeline Sentinel</h1>
        </div>
        <div className="live-indicator">
          <div className="live-dot"></div>
          SYSTEM LIVE
        </div>
      </div>

      <div className="main-content" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        <div className="metrics-grid">
          <div className="glass-panel metric-card">
            <span className="metric-title">Ingestion Rate (orders/min)</span>
            <span className="metric-value">{currentRate}</span>
          </div>
          <div className="glass-panel metric-card">
            <span className="metric-title">Data Quality (Null Rate)</span>
            <span className="metric-value" style={{ color: nullRate > 5 ? 'var(--danger-color)' : 'var(--text-primary)'}}>
              {nullRate}%
            </span>
          </div>
          <div className="glass-panel metric-card">
            <span className="metric-title">Quarantined (1h)</span>
            <span className="metric-value">{metrics.total_quarantined_1h}</span>
          </div>
        </div>

        <div className="glass-panel" style={{ flexGrow: 1 }}>
          <h2 style={{ fontSize: '1rem', marginTop: 0, color: 'var(--text-secondary)'}}>Live Data Stream (PostgreSQL)</h2>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={metrics.chart_data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorOrders" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="var(--accent-color)" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="var(--accent-color)" stopOpacity={0}/>
                  </linearGradient>
                  <linearGradient id="colorNulls" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="var(--danger-color)" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="var(--danger-color)" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
                <XAxis dataKey="time" stroke="var(--text-secondary)" tick={{fontSize: 12}} />
                <YAxis stroke="var(--text-secondary)" tick={{fontSize: 12}} />
                <Tooltip 
                  contentStyle={{ backgroundColor: 'var(--bg-color)', borderColor: 'var(--border-color)' }}
                  itemStyle={{ color: 'var(--text-primary)' }}
                />
                <Area type="monotone" dataKey="orders" stroke="var(--accent-color)" fillOpacity={1} fill="url(#colorOrders)" name="Orders" />
                <Area type="monotone" dataKey="nulls" stroke="var(--danger-color)" fillOpacity={1} fill="url(#colorNulls)" name="Corrupt (Nulls)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="side-panel" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        <div className="glass-panel controls-panel">
          <h2 style={{ fontSize: '1rem', marginTop: 0, color: 'var(--text-secondary)'}}>Chaos Engineering</h2>
          
          <button 
            className={`btn danger ${simState.inject_gap ? 'active' : ''}`}
            onClick={() => toggleAnomaly('gap')}
          >
            <AlertTriangle size={18} />
            {simState.inject_gap ? 'Restore Data Stream' : 'Simulate Outage (Drop)'}
          </button>

          <button 
            className={`btn warning ${simState.inject_nulls ? 'active' : ''}`}
            onClick={() => toggleAnomaly('nulls')}
          >
            <Zap size={18} />
            {simState.inject_nulls ? 'Restore Data Quality' : 'Inject Data Corruption'}
          </button>
          
          <button 
            className={`btn warning ${simState.inject_duplicate ? 'active' : ''}`}
            onClick={() => toggleAnomaly('duplicate')}
          >
            <Zap size={18} />
            {simState.inject_duplicate ? 'Restore Uniqueness' : 'Inject Data Duplication'}
          </button>
        </div>

        <div className="glass-panel" style={{ flexGrow: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
          <h2 style={{ fontSize: '1rem', marginTop: 0, color: 'var(--text-secondary)'}}>Agent Activity Log (LTM)</h2>
          
          <div className="timeline">
            {incidents.length === 0 ? (
              <div style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', textAlign: 'center', marginTop: '2rem' }}>
                <CheckCircle size={32} style={{ marginBottom: '1rem', opacity: 0.5 }} />
                <br/>
                Pipeline is healthy. No incidents recorded.
              </div>
            ) : (
              incidents.map((inc, i) => (
                <div key={i} className={`timeline-item ${inc.severity === 'CRITICAL' ? 'critical' : inc.severity === 'WARNING' ? 'warning' : 'success'}`}>
                  <div className="timeline-time">{new Date(inc.timestamp).toLocaleTimeString()}</div>
                  <div className="timeline-title">{inc.severity} ANOMALY</div>
                  <div className="timeline-desc">
                    {inc.root_cause || "Outage detected."}
                    <br/><br/>
                    <strong style={{color: 'var(--accent-color)'}}>Repair Action:</strong> {inc.resolved === 1 ? (inc.fix_taken || "Resolved manually") : "Pending..."}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
