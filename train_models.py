#!/usr/bin/env python3
"""
Step 3: Model Training
COM747 - Predictive Auto-Scaling Project
Trains: ARIMA (aggregate) + XGBoost + LSTM with Optuna tuning
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             roc_auc_score, precision_score, recall_score,
                             f1_score, accuracy_score)
from sklearn.model_selection import TimeSeriesSplit

RESULTS_DIR = "results"
MODELS_DIR  = "models"
CLEAN_DIR   = "data/clean"
FEATURES_DIR = "data/features"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
train_df = pd.read_parquet(os.path.join(FEATURES_DIR, "train.parquet"))
test_df  = pd.read_parquet(os.path.join(FEATURES_DIR, "test.parquet"))

FEATURE_COLS = [c for c in train_df.columns if c not in [
    'vm_id', 'timestamp', 'datetime', 'min_cpu',
    'target_avg_cpu', 'target_scale_event'
]]

print("=" * 60)
print("STEP 3: MODEL TRAINING")
print("=" * 60)
print(f"Train: {len(train_df):,} rows  |  Test: {len(test_df):,} rows")
print(f"Features: {len(FEATURE_COLS)}")

all_metrics = {}

# ── Helper functions ─────────────────────────────────────────────────────────
def regression_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    return {'rmse': rmse, 'mae': mae}

def classification_metrics(y_true, y_pred_prob):
    y_pred = (y_pred_prob > 0.5).astype(int)
    return {
        'accuracy':  float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall':    float(recall_score(y_true, y_pred, zero_division=0)),
        'f1':        float(f1_score(y_true, y_pred, zero_division=0)),
        'auc':       float(roc_auc_score(y_true, y_pred_prob))
    }

# ── MODEL 1: AGGREGATE ARIMA ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL 1: AGGREGATE ARIMA (Cluster-Level Demand)")
print("=" * 60)
from statsmodels.tsa.arima.model import ARIMA

cluster = pd.read_parquet(os.path.join(CLEAN_DIR, "cluster_demand.parquet"))
train_size  = int(len(cluster) * 0.7)
train_agg   = cluster.iloc[:train_size]
test_agg    = cluster.iloc[train_size:]
print(f"Aggregate series: {len(train_agg)} train, {len(test_agg)} test points")

model_arima  = ARIMA(train_agg['avg_cpu'], order=(2, 1, 2))
fitted_arima = model_arima.fit()
forecast_arima = fitted_arima.forecast(steps=len(test_agg))

arima_reg = regression_metrics(test_agg['avg_cpu'].values, forecast_arima.values)
all_metrics['arima_aggregate_regression'] = arima_reg
print(f"  ARIMA RMSE: {arima_reg['rmse']:.4f}  |  MAE: {arima_reg['mae']:.4f}")

pd.DataFrame({
    'timestamp': test_agg['timestamp'].values,
    'actual':    test_agg['avg_cpu'].values,
    'predicted': forecast_arima.values
}).to_csv(os.path.join(RESULTS_DIR, "arima_predictions.csv"), index=False)

# ── MODEL 2: XGBOOST + WALK-FORWARD VALIDATION ───────────────────────────────
print("\n" + "=" * 60)
print("MODEL 2: XGBOOST + WALK-FORWARD VALIDATION")
print("=" * 60)
from xgboost import XGBRegressor, XGBClassifier

# Walk-forward cross-validation (time-series safe)
tscv = TimeSeriesSplit(n_splits=3)
wf_scores = []
for fold, (train_idx, val_idx) in enumerate(tscv.split(train_df)):
    X_tr  = train_df.iloc[train_idx][FEATURE_COLS]
    y_tr  = train_df.iloc[train_idx]['target_avg_cpu']
    X_val = train_df.iloc[val_idx][FEATURE_COLS]
    y_val = train_df.iloc[val_idx]['target_avg_cpu']

    m = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                     n_jobs=2, random_state=42)
    m.fit(X_tr, y_tr)
    preds = m.predict(X_val)
    rmse  = float(np.sqrt(mean_squared_error(y_val, preds)))
    wf_scores.append(rmse)
    print(f"  Fold {fold+1}: RMSE = {rmse:.4f}")

print(f"  Mean Walk-Forward RMSE: {np.mean(wf_scores):.4f} (+/- {np.std(wf_scores):.4f})")
all_metrics['xgb_walk_forward'] = {
    'mean_rmse': float(np.mean(wf_scores)),
    'std_rmse':  float(np.std(wf_scores))
}

# Final XGBoost regressor
print("  Training final XGBoost regressor...")
xgb_reg = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                        subsample=0.8, n_jobs=2, random_state=42)
xgb_reg.fit(train_df[FEATURE_COLS], train_df['target_avg_cpu'])
xgb_reg_pred = xgb_reg.predict(test_df[FEATURE_COLS])
xgb_reg_metrics = regression_metrics(test_df['target_avg_cpu'].values, xgb_reg_pred)
all_metrics['xgb_regression'] = xgb_reg_metrics

# XGBoost classifier (scale-event detection)
print("  Training XGBoost classifier...")
scale_pos = ((train_df['target_scale_event'] == 0).sum() /
             max(1, (train_df['target_scale_event'] == 1).sum()))
xgb_clf = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                         subsample=0.8, scale_pos_weight=scale_pos,
                         n_jobs=2, random_state=42)
xgb_clf.fit(train_df[FEATURE_COLS], train_df['target_scale_event'])
xgb_clf_prob = xgb_clf.predict_proba(test_df[FEATURE_COLS])[:, 1]
xgb_clf_metrics = classification_metrics(test_df['target_scale_event'].values, xgb_clf_prob)
all_metrics['xgb_classification'] = xgb_clf_metrics

print(f"  XGBoost Reg  RMSE: {xgb_reg_metrics['rmse']:.4f}  MAE: {xgb_reg_metrics['mae']:.4f}")
print(f"  XGBoost Clf  F1:   {xgb_clf_metrics['f1']:.4f}   AUC: {xgb_clf_metrics['auc']:.4f}")

# Save predictions & models
xgb_results = test_df[['vm_id', 'timestamp', 'avg_cpu', 'target_avg_cpu', 'target_scale_event']].copy()
xgb_results['predicted_cpu']         = xgb_reg_pred
xgb_results['predicted_scale_prob']  = xgb_clf_prob
xgb_results['predicted_scale']       = (xgb_clf_prob > 0.5).astype(int)
xgb_results.to_csv(os.path.join(RESULTS_DIR, "xgb_predictions.csv"), index=False)
xgb_reg.save_model(os.path.join(MODELS_DIR, "xgb_regressor.json"))
xgb_clf.save_model(os.path.join(MODELS_DIR, "xgb_classifier.json"))
print("  [SAVED] XGBoost models")

# ── MODEL 3: LSTM + OPTUNA ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL 3: LSTM + OPTUNA HYPERPARAMETER TUNING")
print("=" * 60)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")


def create_sequences(df, seq_len, feature_cols, target_col):
    X, y = [], []
    for vm_id, group in df.groupby('vm_id'):
        group = group.sort_values('timestamp').reset_index(drop=True)
        if len(group) < seq_len:
            continue
        vals  = group[feature_cols].values.astype(np.float32)
        targs = group[target_col].values.astype(np.float32)
        for i in range(seq_len - 1, len(group)):
            X.append(vals[i - seq_len + 1: i + 1])
            y.append(targs[i])
    return np.array(X), np.array(y)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc1     = nn.Linear(hidden_size, 32)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2     = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        return self.fc2(out).squeeze()


def make_loaders(seq_len, batch):
    X_tr, y_tr   = create_sequences(train_df, seq_len, FEATURE_COLS, 'target_avg_cpu')
    X_val, y_val = create_sequences(test_df,  seq_len, FEATURE_COLS, 'target_avg_cpu')
    tr_ds  = torch.utils.data.TensorDataset(torch.FloatTensor(X_tr),  torch.FloatTensor(y_tr))
    val_ds = torch.utils.data.TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    return (DataLoader(tr_ds,  batch_size=batch, shuffle=True),
            DataLoader(val_ds, batch_size=batch),
            X_val, y_val)


def objective(trial):
    hidden  = trial.suggest_categorical('hidden',   [32, 64])
    layers  = trial.suggest_int('layers',            1, 2)
    dropout = trial.suggest_float('dropout',         0.1, 0.3)
    lr      = trial.suggest_float('lr',              1e-4, 1e-2, log=True)
    batch   = trial.suggest_categorical('batch',    [64, 128])
    seq_len = trial.suggest_categorical('seq_len',  [3, 5])

    train_loader, val_loader, X_val, y_val = make_loaders(seq_len, batch)
    if len(X_val) == 0:
        return float('inf')

    model     = LSTMModel(len(FEATURE_COLS), hidden, layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(5):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            optimizer.step()

    model.eval()
    val_losses = []
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
            val_losses.append(criterion(model(bx), by).item())
    return float(np.mean(val_losses))


print("  Running Optuna (20 trials) — this is the slow part, please wait...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)
best = study.best_params
print(f"  Best trial MSE: {study.best_value:.4f}")
print(f"  Best params:    {best}")

# Train final model with best hyperparameters
print("  Training final LSTM...")
train_loader, val_loader, X_val, y_val = make_loaders(best['seq_len'], best['batch'])

final_model = LSTMModel(len(FEATURE_COLS), best['hidden'], best['layers'], best['dropout']).to(device)
optimizer   = torch.optim.Adam(final_model.parameters(), lr=best['lr'])
criterion   = nn.MSELoss()
best_val    = float('inf')

for epoch in range(20):
    final_model.train()
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        criterion(final_model(bx), by).backward()
        optimizer.step()

    final_model.eval()
    val_losses = []
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
            val_losses.append(criterion(final_model(bx), by).item())
    v = float(np.mean(val_losses))
    if v < best_val:
        best_val = v
        torch.save(final_model.state_dict(), os.path.join(MODELS_DIR, "lstm_best.pt"))
    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1}/20: Val Loss = {v:.4f}")

# Evaluate LSTM
final_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "lstm_best.pt")))
final_model.eval()
lstm_preds, lstm_trues = [], []
with torch.no_grad():
    for bx, by in val_loader:
        bx = bx.to(device)
        lstm_preds.extend(final_model(bx).cpu().numpy())
        lstm_trues.extend(by.numpy())

lstm_reg  = regression_metrics(np.array(lstm_trues), np.array(lstm_preds))
lstm_prob = np.clip(np.array(lstm_preds) / 100, 0, 1)
lstm_clf  = classification_metrics((np.array(lstm_trues) > 75).astype(int), lstm_prob)
all_metrics['lstm_regression']     = lstm_reg
all_metrics['lstm_classification'] = lstm_clf
print(f"  LSTM RMSE: {lstm_reg['rmse']:.4f}  F1: {lstm_clf['f1']:.4f}  AUC: {lstm_clf['auc']:.4f}")

pd.DataFrame({
    'actual_cpu':      lstm_trues,
    'predicted_cpu':   lstm_preds,
    'actual_scale':    (np.array(lstm_trues) > 75).astype(int),
    'predicted_scale': (np.array(lstm_preds) > 75).astype(int)
}).to_csv(os.path.join(RESULTS_DIR, "lstm_predictions.csv"), index=False)

# ── DRIFT DETECTOR ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DRIFT DETECTION")
print("=" * 60)

class DriftDetector:
    def __init__(self, threshold=3.0, window=50):
        self.threshold = threshold
        self.window    = window
        self.errors    = []
        self.drift_count = 0

    def update(self, actual, predicted):
        error = abs(actual - predicted)
        self.errors.append(error)
        if len(self.errors) > self.window:
            self.errors.pop(0)
        if len(self.errors) >= 10:
            mean_err = np.mean(self.errors)
            std_err  = np.std(self.errors)
            if error > mean_err + self.threshold * std_err:
                self.drift_count += 1
                return True
        return False


detector = DriftDetector()
drift_events = 0
for actual, predicted in zip(test_df['target_avg_cpu'].values[:500],
                              xgb_reg_pred[:500]):
    if detector.update(actual, predicted):
        drift_events += 1

print(f"  Drift events detected in first 500 predictions: {drift_events}")
all_metrics['drift_detection'] = {'drift_events_500': drift_events}

# ── Save all metrics ─────────────────────────────────────────────────────────
with open(os.path.join(RESULTS_DIR, "all_metrics.json"), 'w') as f:
    json.dump(all_metrics, f, indent=2)
print(f"\n[SAVED] All metrics → {RESULTS_DIR}/all_metrics.json")

print("\n" + "=" * 60)
print("TRAINING COMPLETE — RESULTS SUMMARY")
print("=" * 60)
print(f"  ARIMA RMSE:    {all_metrics['arima_aggregate_regression']['rmse']:.4f}")
print(f"  XGBoost RMSE:  {xgb_reg_metrics['rmse']:.4f}  F1: {xgb_clf_metrics['f1']:.4f}  AUC: {xgb_clf_metrics['auc']:.4f}")
print(f"  LSTM    RMSE:  {lstm_reg['rmse']:.4f}  F1: {lstm_clf['f1']:.4f}  AUC: {lstm_clf['auc']:.4f}")
print(f"  Walk-Forward:  {all_metrics['xgb_walk_forward']['mean_rmse']:.4f} +/- {all_metrics['xgb_walk_forward']['std_rmse']:.4f}")
