#!/usr/bin/env python3
"""
Step 5: Build Standalone HTML Dashboard
COM747 - Predictive Auto-Scaling Project
Generates dashboard.html with all 10 visualisations embedded
Run from the project root: python build_dashboard.py
"""
import json
import os

VIZ_DIR = "visualisations"
RESULTS_DIR = "results"
OUT_FILE = "dashboard.html"

# Load all JSON data files
files = {
    "roc_curves": "roc_curves.json",
    "threshold_sensitivity": "threshold_sensitivity.json",
    "residual_analysis": "residual_analysis.json",
    "calibration": "calibration.json",
    "prediction_scatter": "prediction_scatter.json",
    "temporal_error": "temporal_error.json",
    "simulation_results": "simulation_results.json",
    "decision_timeline": "decision_timeline.json",
    "model_comparison": "model_comparison.json",
    "walk_forward": "walk_forward.json",
}

embedded = {}
for key, filename in files.items():
    path = os.path.join(VIZ_DIR, filename)
    if os.path.exists(path):
        with open(path, 'r') as f:
            embedded[key] = json.load(f)
        print(f"  [OK] Loaded {filename}")
    else:
        print(f"  [SKIP] {filename} not found - run simulate.py first")
        embedded[key] = {}

# Pull real metric-card values from the actual run instead of hardcoding them
metrics_path = os.path.join(RESULTS_DIR, "all_metrics.json")
sim_path = os.path.join(RESULTS_DIR, "simulation_summary.json")

xgb_auc = xgb_f1 = lstm_f1 = cost_reduction = sla_reduction = "N/A"
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        m = json.load(f)
    xgb_auc = f"{m['xgb_classification']['auc']:.3f}"
    xgb_f1 = f"{m['xgb_classification']['f1']:.3f}"
    lstm_f1 = f"{m['lstm_classification']['f1']:.3f}"

if os.path.exists(sim_path):
    with open(sim_path) as f:
        s = json.load(f)
    cost_reduction = f"{s['cost_reduction_pct']:.1f}%"
    sla_reduction = f"{s['sla_reduction_pct']:.1f}%"

# Real confusion-matrix numbers for the reactive vs predictive comparison card
reactive_fn = predictive_fn = reactive_cost = predictive_cost = "N/A"
if os.path.exists(sim_path):
    reactive_fn = s['reactive']['fn']
    predictive_fn = s['predictive']['fn']
    reactive_cost = f"${s['reactive']['cost']:.2f}"
    predictive_cost = f"${s['predictive']['cost']:.2f}"

data_js = f"const DATA = {json.dumps(embedded)};"

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Predictive Auto-Scaling | 10 Visualisations | COM747</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0f172a; color: #e2e8f0; }}
header {{ background: linear-gradient(135deg, #1e40af, #7c3aed); padding: 2rem; text-align: center; }}
header h1 {{ font-size: 1.8rem; font-weight: 700; }}
header p {{ margin-top: 0.4rem; opacity: 0.85; font-size: 0.95rem; }}
.metrics {{ display: flex; justify-content: center; gap: 1.5rem; padding: 1.5rem; flex-wrap: wrap; }}
.metric-card {{ background: #1e293b; border-radius: 10px; padding: 1rem 1.5rem; text-align: center; min-width: 140px; }}
.metric-card .value {{ font-size: 1.8rem; font-weight: 700; color: #38bdf8; }}
.metric-card .label {{ font-size: 0.75rem; color: #94a3b8; margin-top: 0.25rem; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 1.5rem; padding: 1.5rem; }}
.card {{ background: #1e293b; border-radius: 12px; padding: 1.25rem; }}
.card h2 {{ font-size: 0.95rem; font-weight: 600; color: #94a3b8; margin-bottom: 1rem; text-transform: uppercase; letter-spacing: 0.05em; }}
canvas {{ max-height: 280px; }}
.sim-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
.sim-box {{ background: #0f172a; border-radius: 8px; padding: 1rem; text-align: center; }}
.sim-box.reactive {{ border: 2px solid #ef4444; }}
.sim-box.predictive {{ border: 2px solid #22c55e; }}
.sim-box h3 {{ font-size: 0.8rem; margin-bottom: 0.5rem; }}
.sim-box .big {{ font-size: 1.4rem; font-weight: 700; }}
.sim-box.reactive .big {{ color: #ef4444; }}
.sim-box.predictive .big {{ color: #22c55e; }}
footer {{ text-align: center; padding: 2rem; color: #475569; font-size: 0.8rem; }}
</style>
</head>
<body>
<header>
  <h1>Predictive Auto-Scaling Dashboard</h1>
  <p>COM747 Data Science &amp; Machine Learning &nbsp;|&nbsp; Ulster University &nbsp;|&nbsp; Azure VM Traces</p>
</header>

<div class="metrics">
  <div class="metric-card"><div class="value">{xgb_auc}</div><div class="label">XGBoost AUC-ROC</div></div>
  <div class="metric-card"><div class="value">{xgb_f1}</div><div class="label">XGBoost F1 Score</div></div>
  <div class="metric-card"><div class="value">{lstm_f1}</div><div class="label">LSTM F1 Score</div></div>
  <div class="metric-card"><div class="value">{cost_reduction}</div><div class="label">Cost Reduction</div></div>
  <div class="metric-card"><div class="value">{sla_reduction}</div><div class="label">SLA Improvement</div></div>
  <div class="metric-card"><div class="value">~3ms</div><div class="label">API Latency</div></div>
</div>

<div class="grid">
  <div class="card"><h2>1. ROC Curves</h2><canvas id="roc"></canvas></div>
  <div class="card"><h2>2. Threshold Sensitivity</h2><canvas id="threshold"></canvas></div>
  <div class="card"><h2>3. Residual Analysis</h2><canvas id="residual"></canvas></div>
  <div class="card"><h2>4. Calibration Plot</h2><canvas id="calib"></canvas></div>
  <div class="card"><h2>5. Predicted vs Actual CPU</h2><canvas id="scatter"></canvas></div>
  <div class="card"><h2>6. Temporal Error Heatmap</h2><canvas id="heatmap"></canvas></div>
  <div class="card">
    <h2>7. Reactive vs Predictive Policy</h2>
    <div class="sim-grid">
      <div class="sim-box reactive">
        <h3>REACTIVE BASELINE</h3>
        <div class="big" style="color:#ef4444" id="r-violations">{reactive_fn}</div>
        <div style="font-size:0.75rem;color:#94a3b8">SLA Violations &nbsp;|&nbsp; {reactive_cost}</div>
      </div>
      <div class="sim-box predictive">
        <h3>PREDICTIVE POLICY</h3>
        <div class="big" style="color:#22c55e" id="p-violations">{predictive_fn}</div>
        <div style="font-size:0.75rem;color:#94a3b8">SLA Violations &nbsp;|&nbsp; {predictive_cost}</div>
      </div>
    </div>
    <canvas id="confmat" style="margin-top:1rem"></canvas>
  </div>
  <div class="card"><h2>8. Decision Timeline (Single VM)</h2><canvas id="timeline"></canvas></div>
  <div class="card"><h2>9. Model Comparison</h2><canvas id="comparison"></canvas>
    <p style="font-size:0.75rem;color:#94a3b8;margin-top:0.5rem">ARIMA forecasts cluster-mean CPU (aggregate), so its RMSE is not directly comparable and it has no F1/AUC.</p></div>
  <div class="card"><h2>10. Walk-Forward Validation</h2><canvas id="walkfwd"></canvas></div>
</div>

<footer>COM747 Group Project &mdash; Ulster University &mdash; August 2026</footer>

<script>
{data_js}

const C = (id, type, data, opts={{}}) => new Chart(document.getElementById(id), {{
  type, data, options: {{responsive:true, plugins:{{legend:{{labels:{{color:'#e2e8f0'}}}}}}, ...opts}}
}});

// 1. ROC
if (DATA.roc_curves && DATA.roc_curves.xgboost) {{
  C('roc', 'line', {{
    datasets: [
      {{label: 'XGBoost (AUC=' + DATA.roc_curves.xgboost.auc.toFixed(3) + ')', data: DATA.roc_curves.xgboost.fpr.map((x,i)=>({{x, y:DATA.roc_curves.xgboost.tpr[i]}})), borderColor:'#38bdf8', fill:false, pointRadius:0}},
      {{label: 'LSTM (AUC=' + DATA.roc_curves.lstm.auc.toFixed(3) + ')', data: DATA.roc_curves.lstm.fpr.map((x,i)=>({{x, y:DATA.roc_curves.lstm.tpr[i]}})), borderColor:'#a78bfa', fill:false, pointRadius:0}},
      {{label: 'Random', data: [{{x:0,y:0}},{{x:1,y:1}}], borderColor:'#475569', borderDash:[5,5], fill:false, pointRadius:0}}
    ]
  }}, {{scales:{{x:{{title:{{display:true,text:'False Positive Rate',color:'#94a3b8'}},ticks:{{color:'#94a3b8'}}}},y:{{title:{{display:true,text:'True Positive Rate',color:'#94a3b8'}},ticks:{{color:'#94a3b8'}}}}}}}});
}}

// 2. Threshold sensitivity
if (DATA.threshold_sensitivity && DATA.threshold_sensitivity.length) {{
  const ts = DATA.threshold_sensitivity;
  C('threshold', 'line', {{
    labels: ts.map(d=>d.threshold+'%'),
    datasets: [
      {{label:'Total Cost ($)', data: ts.map(d=>d.total_cost), borderColor:'#f59e0b', tension:0.3, yAxisID:'y'}},
      {{label:'SLA Violations', data: ts.map(d=>d.sla_violations), borderColor:'#ef4444', tension:0.3, yAxisID:'y1'}}
    ]
  }}, {{scales:{{x:{{ticks:{{color:'#94a3b8'}}}},y:{{ticks:{{color:'#94a3b8'}}}},y1:{{position:'right',ticks:{{color:'#ef4444'}}}}}}}});
}}

// 3. Residuals histogram
if (DATA.residual_analysis && DATA.residual_analysis.residuals) {{
  const bins = Array.from({{length:20}}, (_,i)=> -20 + i*2);
  const counts = bins.map(b => DATA.residual_analysis.residuals.filter(r=>r>=b && r<b+2).length);
  C('residual', 'bar', {{
    labels: bins.map(b=>b.toFixed(0)),
    datasets: [{{label:'Residuals', data:counts, backgroundColor:'#38bdf8aa'}}]
  }}, {{scales:{{x:{{ticks:{{color:'#94a3b8'}},title:{{display:true,text:'Residual (CPU%)',color:'#94a3b8'}}}},y:{{ticks:{{color:'#94a3b8'}}}}}}}});
}}

// 4. Calibration
if (DATA.calibration && DATA.calibration.length) {{
  C('calib', 'bar', {{
    labels: DATA.calibration.map(d=>d.bin_mid.toFixed(2)),
    datasets: [
      {{label:'Actual Rate', data: DATA.calibration.map(d=>d.actual_rate), backgroundColor:'#22c55eaa'}},
      {{label:'Predicted Rate', data: DATA.calibration.map(d=>d.mean_predicted), backgroundColor:'#38bdf8aa'}}
    ]
  }}, {{scales:{{x:{{ticks:{{color:'#94a3b8'}}}},y:{{ticks:{{color:'#94a3b8'}}}}}}}});
}}

// 5. Scatter
if (DATA.prediction_scatter && DATA.prediction_scatter.actual) {{
  const pts = DATA.prediction_scatter.actual.map((a,i)=>({{x:a, y:DATA.prediction_scatter.predicted[i]}}));
  C('scatter', 'scatter', {{
    datasets: [
      {{label:'Predictions', data:pts, backgroundColor:'#38bdf855', pointRadius:3}},
      {{label:'Perfect', data:[{{x:0,y:0}},{{x:100,y:100}}], type:'line', borderColor:'#ef4444', pointRadius:0, fill:false}}
    ]
  }}, {{scales:{{x:{{title:{{display:true,text:'Actual CPU%',color:'#94a3b8'}},ticks:{{color:'#94a3b8'}}}},y:{{title:{{display:true,text:'Predicted CPU%',color:'#94a3b8'}},ticks:{{color:'#94a3b8'}}}}}}}});
}}

// 6. Temporal heatmap (as bar)
if (DATA.temporal_error && DATA.temporal_error.length) {{
  C('heatmap', 'bar', {{
    labels: DATA.temporal_error.map(d=>d.time_bucket),
    datasets: [{{label:'Mean Abs Error', data:DATA.temporal_error.map(d=>d.mean_error), backgroundColor:'#f59e0baa'}}]
  }}, {{scales:{{x:{{title:{{display:true,text:'5-min time bucket',color:'#94a3b8'}},ticks:{{color:'#94a3b8'}}}},y:{{ticks:{{color:'#94a3b8'}}}}}}}});
}}

// 7. Confusion matrix bar (real numbers from simulation_results.json)
if (DATA.simulation_results) {{
  const sr = DATA.simulation_results;
  C('confmat', 'bar', {{
    labels: ['True Positives','False Positives','False Negatives'],
    datasets: [
      {{label:'Reactive', data:[sr.reactive.tp, sr.reactive.fp, sr.reactive.fn], backgroundColor:'#ef444488'}},
      {{label:'Predictive', data:[sr.predictive.tp, sr.predictive.fp, sr.predictive.fn], backgroundColor:'#22c55e88'}}
    ]
  }}, {{scales:{{x:{{ticks:{{color:'#94a3b8'}}}},y:{{ticks:{{color:'#94a3b8'}}}}}}}});
}}

// 8. Decision timeline
if (DATA.decision_timeline && DATA.decision_timeline.length) {{
  const tl = DATA.decision_timeline;
  C('timeline', 'line', {{
    labels: tl.map((_,i)=>i),
    datasets: [
      {{label:'Actual CPU', data:tl.map(d=>d.actual), borderColor:'#38bdf8', fill:false, tension:0.3}},
      {{label:'Predicted CPU', data:tl.map(d=>d.predicted), borderColor:'#a78bfa', borderDash:[4,4], fill:false, tension:0.3}},
      {{label:'Target (15m)', data:tl.map(d=>d.target), borderColor:'#22c55e', fill:false, tension:0.3}}
    ]
  }}, {{scales:{{x:{{ticks:{{color:'#94a3b8'}}}},y:{{ticks:{{color:'#94a3b8'}},title:{{display:true,text:'CPU %',color:'#94a3b8'}}}}}}}});
}}

// 9. Model comparison
if (DATA.model_comparison && DATA.model_comparison.models) {{
  const mc = DATA.model_comparison;
  C('comparison', 'bar', {{
    labels: mc.models,
    datasets: [
      {{label:'RMSE', data:mc.rmse, backgroundColor:'#ef444488'}},
      {{label:'F1', data:mc.f1, backgroundColor:'#22c55e88'}},
      {{label:'AUC', data:mc.auc, backgroundColor:'#38bdf888'}}
    ]
  }}, {{scales:{{x:{{ticks:{{color:'#94a3b8'}}}},y:{{ticks:{{color:'#94a3b8'}}}}}}}});
}}

// 10. Walk-forward
if (DATA.walk_forward && DATA.walk_forward.mean_rmse) {{
  const wf = DATA.walk_forward;
  C('walkfwd', 'bar', {{
    labels: [...(wf.fold_rmse || []).map((_,i)=>'Fold '+(i+1)), 'Mean'],
    datasets: [{{
      label:'RMSE',
      data:[...(wf.fold_rmse || []), wf.mean_rmse],
      backgroundColor:[...(wf.fold_rmse || []).map(()=>'#38bdf888'), '#f59e0bcc']
    }}]
  }}, {{scales:{{x:{{ticks:{{color:'#94a3b8'}}}},y:{{ticks:{{color:'#94a3b8'}}}}}}}});
}}
</script>
</body>
</html>"""

with open(OUT_FILE, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\n[DONE] Dashboard saved to {OUT_FILE}")
print("Open dashboard.html in Chrome to view all 10 visualisations")
