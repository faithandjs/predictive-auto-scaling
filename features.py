#!/usr/bin/env python3
"""
Step 2: Feature Engineering (MEMORY-SAFE VERSION)
COM747 - Predictive Auto-Scaling Project
Builds 19 features per VM per timestep
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

CLEAN_DIR = "data/clean"
FEATURES_DIR = "data/features"
os.makedirs(FEATURES_DIR, exist_ok=True)

SEQ_LENGTH = 5
FORECAST_HORIZON = 3  # predict 15 mins ahead (3 x 5-min intervals)
MAX_VMS = 5000  # CRITICAL: limits RAM usage to ~8GB

print("=" * 60)
print("STEP 2: FEATURE ENGINEERING")
print("=" * 60)

df = pd.read_parquet(os.path.join(CLEAN_DIR, "azure_vm_clean.parquet"))
print(f"Loaded: {len(df):,} rows, {df['vm_id'].nunique():,} VMs")

# SAMPLE DOWN TO PREVENT OUT-OF-MEMORY CRASH
if df['vm_id'].nunique() > MAX_VMS:
    print(f"Sampling down to {MAX_VMS} VMs to fit in 8GB RAM...")
    selected_vms = df['vm_id'].drop_duplicates().sample(n=MAX_VMS, random_state=42).values
    df = df[df['vm_id'].isin(selected_vms)].copy()
    print(f"After sampling: {len(df):,} rows, {df['vm_id'].nunique():,} VMs")

df = df.sort_values(['vm_id', 'timestamp']).reset_index(drop=True)

all_features = []
total_vms = df['vm_id'].nunique()
for idx, (vm_id, group) in enumerate(df.groupby('vm_id')):
    if idx % 500 == 0:
        print(f" Processing VM {idx}/{total_vms}...")

    group = group.copy().sort_values('timestamp').reset_index(drop=True)
    if len(group) < SEQ_LENGTH + FORECAST_HORIZON + 1:
        continue

    # Lag features (past values)
    for lag in [1, 2, 3, 5]:
        group[f'avg_cpu_lag_{lag}'] = group['avg_cpu'].shift(lag)
        group[f'max_cpu_lag_{lag}'] = group['max_cpu'].shift(lag)

    # Rolling statistics (trends)
    for window in [3, 5]:
        group[f'avg_cpu_roll_mean_{window}'] = group['avg_cpu'].shift(1).rolling(window=window).mean()
        group[f'avg_cpu_roll_std_{window}'] = group['avg_cpu'].shift(1).rolling(window=window).std()
        group[f'max_cpu_roll_mean_{window}'] = group['max_cpu'].shift(1).rolling(window=window).mean()

        # Delta features (rate of change)
        group['avg_cpu_delta_1'] = group['avg_cpu'] - group['avg_cpu_lag_1']
        group['avg_cpu_delta_3'] = group['avg_cpu'] - group['avg_cpu_lag_3']

        # Temporal feature
        group['time_of_day'] = group['timestamp'] % 86400

        # Target: CPU 15 minutes ahead
        group['target_avg_cpu'] = group['avg_cpu'].shift(-FORECAST_HORIZON)
        group['target_scale_event'] = (group['target_avg_cpu'] > 75).astype(int)

        all_features.append(group)

features_df = pd.concat(all_features, ignore_index=True)
features_df = features_df.dropna()
print(f"\nFeatures built: {len(features_df):,} rows")

# Time-based train/test split (no random shuffle for time-series!)
split_timestamp = features_df['timestamp'].quantile(0.7)
train_df = features_df[features_df['timestamp'] <= split_timestamp].copy()
test_df = features_df[features_df['timestamp'] > split_timestamp].copy()
print(f"Train: {len(train_df):,} rows ({train_df['vm_id'].nunique():,} VMs)")
print(f"Test: {len(test_df):,} rows ({test_df['vm_id'].nunique():,} VMs)")

# Columns to scale (exclude IDs, timestamps, targets)
feature_cols = [c for c in train_df.columns if c not in [
    'vm_id', 'timestamp', 'datetime', 'min_cpu',
    'target_avg_cpu', 'target_scale_event', 'time_of_day'
]]

# Fit scaler ONLY on training data (prevent data leakage)
scaler = StandardScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
test_df[feature_cols] = scaler.transform(test_df[feature_cols])

# Save scaler for use in inference API
with open(os.path.join(FEATURES_DIR, "scaler.pkl"), 'wb') as f:
    pickle.dump(scaler, f)

train_df.to_parquet(os.path.join(FEATURES_DIR, "train.parquet"), index=False)
test_df.to_parquet(os.path.join(FEATURES_DIR, "test.parquet"), index=False)

print(f"\n[DONE] Features saved to {FEATURES_DIR}/")
print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
