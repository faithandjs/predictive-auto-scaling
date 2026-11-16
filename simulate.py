#!/usr/bin/env python3
"""
Step 4: Cost-SLA Simulation + Visualisation Data Generation
COM747 - Predictive Auto-Scaling Project
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


RESULTS_DIR = "results"
VIZ_DIR     = "visualisations"
os.makedirs(VIZ_DIR, exist_ok=True)


print("=" * 60)
print("STEP 4: SIMULATION & VISUALISATION DATA")
print("=" * 60)


# Load data
xgb = pd.read_csv(os.path.join(RESULTS_DIR, "xgb_predictions.csv"))
lstm = pd.read_csv(os.path.join(RESULTS_DIR, "lstm_predictions.csv"))
with open(os.path.join(RESULTS_DIR, "all_metrics.json")) as f:
    metrics = json.load(f)


# COST-SLA SIMULATION
SCALE_UP_COST    = 0.10   # cost per unnecessary scale-up ($)
SLA_VIOLATION_COST = 1.00 # cost per missed overload ($)
THRESHOLD        = 75.0


print("\nRunning cost-SLA simulation...")


reactive_scale = (xgb['avg_cpu'] > THRESHOLD).astype(int)
predictive_scale = xgb['predicted_scale']
actual          = xgb['target_scale_event']


# Reactive: scales only when currently over threshold
reactive_tp = int(((reactive_scale == 1) & (actual == 1)).sum())
reactive_fp = int(((reactive_scale == 1) & (actual == 0)).sum())
reactive_fn = int(((reactive_scale == 0) & (actual == 1)).sum())
reactive_cost = reactive_fp * SCALE_UP_COST + reactive_fn * SLA_VIOLATION_COST


# Predictive: scales based on 15-min-ahead forecast
pred_tp = int(((predictive_scale == 1) & (actual == 1)).sum())
pred_fp = int(((predictive_scale == 1) & (actual == 0)).sum())
pred_fn = int(((predictive_scale == 0) & (actual == 1)).sum())
pred_cost = pred_fp * SCALE_UP_COST + pred_fn * SLA_VIOLATION_COST


print(f"\n REACTIVE POLICY: TP={reactive_tp}, FP={reactive_fp}, FN={reactive_fn} Cost=${reactive_cost:.2f}")
print(f" PREDICTIVE POLICY: TP={pred_tp},     FP={pred_fp},     FN={pred_fn} Cost=${pred_cost:.2f}")
print(f"\n Cost reduction: {(1 - pred_cost/max(reactive_cost,1))*100:.1f}%")
print(f"   SLA reduction: {(1 - pred_fn/max(reactive_fn,1))*100:.1f}%")


simulation_summary = {
    "reactive":   {"tp": reactive_tp, "fp": reactive_fp, "fn": reactive_fn, "cost":
round(reactive_cost, 2)},
    "predictive": {"tp": pred_tp,     "fp": pred_fp,     "fn": pred_fn,     "cost":
round(pred_cost, 2)},
    "cost_reduction_pct": round((1 - pred_cost/max(reactive_cost, 1))*100, 1),
    "sla_reduction_pct": round((1 - pred_fn/max(reactive_fn, 1))*100, 1)
}
with open(os.path.join(RESULTS_DIR, "simulation_summary.json"), 'w') as f:
    json.dump(simulation_summary, f, indent=2)


# GENERATE VISUALISATION DATA
print("\nGenerating visualisation datasets...")


# 1. ROC curves
from sklearn.metrics import roc_curve
fpr_x, tpr_x, _ = roc_curve(xgb['target_scale_event'], xgb['predicted_scale_prob'])
lstm_prob        = np.clip(lstm['predicted_cpu'] / 100, 0, 1)
fpr_l, tpr_l, _ = roc_curve(lstm['actual_scale'], lstm_prob)
roc_data = {
    "xgboost": {"fpr": fpr_x.tolist()[::10], "tpr": tpr_x.tolist()[::10],
                "auc": metrics['xgb_classification']['auc']},
    "lstm":    {"fpr": fpr_l.tolist()[::10], "tpr": tpr_l.tolist()[::10],
                "auc": metrics['lstm_classification']['auc']}
}
with open(os.path.join(VIZ_DIR, "roc_curves.json"), 'w') as f:
    json.dump(roc_data, f)
print(" 1. ROC curves saved")


# 2. Threshold sensitivity
thresholds = np.arange(50, 96, 5)
sensitivity = []
for t in thresholds:
    ps = (xgb['predicted_scale_prob'] > t/100).astype(int)
    fp = int(((ps == 1) & (actual == 0)).sum())
    fn = int(((ps == 0) & (actual == 1)).sum())
    sensitivity.append({
        "threshold": int(t),
        "scale_events": fp,
        "sla_violations": fn,
        "total_cost": round(fp * SCALE_UP_COST + fn * SLA_VIOLATION_COST, 2)
    })
with open(os.path.join(VIZ_DIR, "threshold_sensitivity.json"), 'w') as f:
    json.dump(sensitivity, f)
print(" 2. Threshold sensitivity saved")


# 3. Residual analysis
residuals = xgb['target_avg_cpu'].values - xgb['predicted_cpu'].values
residual_data = {
    "residuals": residuals[:2000].tolist(),
    "predicted": xgb['predicted_cpu'].values[:2000].tolist(),
    "mean":       float(np.mean(residuals)),
    "std":        float(np.std(residuals)),
    "mae":        float(np.mean(np.abs(residuals)))
}
with open(os.path.join(VIZ_DIR, "residual_analysis.json"), 'w') as f:
    json.dump(residual_data, f)

print("   3. Residual analysis saved")


# 4. Calibration plot
prob_bins = np.linspace(0, 1, 11)
calib = []
for lo, hi in zip(prob_bins[:-1], prob_bins[1:]):
    mask = (xgb['predicted_scale_prob'] >= lo) & (xgb['predicted_scale_prob'] < hi)
    if mask.sum() > 0:
        calib.append({
            "bin_mid":       round((lo + hi) / 2, 2),
            "mean_predicted": round(xgb.loc[mask, 'predicted_scale_prob'].mean(), 4),
            "actual_rate":    round(xgb.loc[mask, 'target_scale_event'].mean(), 4),
            "count":          int(mask.sum())
        })
with open(os.path.join(VIZ_DIR, "calibration.json"), 'w') as f:
    json.dump(calib, f)
print(" 4. Calibration plot saved")


# 5. Prediction scatter (sample 500)
sample = xgb.sample(500, random_state=42)
scatter = {
    "actual":    sample['target_avg_cpu'].tolist(),
    "predicted": sample['predicted_cpu'].tolist()
}
with open(os.path.join(VIZ_DIR, "prediction_scatter.json"), 'w') as f:
    json.dump(scatter, f)
print(" 5. Prediction scatter saved")


# 6. Temporal error heatmap
xgb['abs_error'] = np.abs(xgb['target_avg_cpu'] - xgb['predicted_cpu'])
xgb['time_bucket'] = (xgb['timestamp'] % 86400 // 300).astype(int)
heatmap = xgb.groupby('time_bucket')['abs_error'].mean().reset_index()
with open(os.path.join(VIZ_DIR, "temporal_error.json"), 'w') as f:
    json.dump(heatmap.rename(columns={'abs_error': 'mean_error'}).to_dict('records'), f)
print(" 6. Temporal error heatmap saved")


# 7. Simulation / confusion matrices
cm_r = confusion_matrix(actual, reactive_scale).tolist()
cm_p = confusion_matrix(actual, predictive_scale).tolist()
sim_viz = {
    "reactive":   {"tn": cm_r[0][0], "fp": cm_r[0][1], "fn": cm_r[1][0], "tp": cm_r[1][1]},
    "predictive": {"tn": cm_p[0][0], "fp": cm_p[0][1], "fn": cm_p[1][0], "tp": cm_p[1][1]},
    "savings":    simulation_summary
}
with open(os.path.join(VIZ_DIR, "simulation_results.json"), 'w') as f:
    json.dump(sim_viz, f)
print(" 7. Simulation results saved")


# 8. Decision timeline (single VM)
top_vm = xgb['vm_id'].value_counts().index[0]
vm_data = xgb[xgb['vm_id'] == top_vm].sort_values('timestamp').head(40)
timeline = [{
    "timestamp":          int(r['timestamp']),
    "actual":             round(r['avg_cpu'], 2),
    "predicted":          round(r['predicted_cpu'], 2),
    "target":             round(r['target_avg_cpu'], 2),
    "predicted_decision": int(r['predicted_cpu'] > THRESHOLD),
    "actual_decision":    int(r['target_avg_cpu'] > THRESHOLD)
} for _, r in vm_data.iterrows()]

with open(os.path.join(VIZ_DIR, "decision_timeline.json"), 'w') as f:
    json.dump(timeline, f)
print(" 8. Decision timeline saved")


# 9. Model comparison
comparison = {
    "models": ["ARIMA (Aggregate)", "XGBoost", "LSTM (Optuna)"],
    "rmse": [metrics['arima_aggregate_regression']['rmse'],
             metrics['xgb_regression']['rmse'],
             metrics['lstm_regression']['rmse']],
    # ARIMA forecasts cluster-level demand only, so it has no per-VM
    # scale-event classification: F1/AUC are not applicable (null)
    "f1":   [None, metrics['xgb_classification']['f1'],
             metrics['lstm_classification']['f1']],
    "auc":  [None, metrics['xgb_classification']['auc'],
             metrics['lstm_classification']['auc']],
    "note": "ARIMA RMSE is on cluster-mean CPU %, not per-VM CPU %"
}
with open(os.path.join(VIZ_DIR, "model_comparison.json"), 'w') as f:
    json.dump(comparison, f)
print(" 9. Model comparison saved")


# 10. Walk-forward validation scores
with open(os.path.join(VIZ_DIR, "walk_forward.json"), 'w') as f:
    json.dump(metrics['xgb_walk_forward'], f)
print(" 10. Walk-forward scores saved")


print(f"\n[DONE] All visualisation data saved to {VIZ_DIR}/")
print("\nNext: run python build_dashboard.py to generate the HTML dashboard")
