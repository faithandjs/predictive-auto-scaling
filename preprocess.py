#!/usr/bin/env python3
"""
Step 1: Download and clean Azure VM traces (file 1 of 195)
COM747 - Predictive Auto-Scaling Project
"""
import os
import urllib.request
import pandas as pd
import numpy as np


RAW_DIR = "data/raw"
CLEAN_DIR = "data/clean"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)


READINGS_URL = (
    "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-v2/"
    "trace_data_vm_cpu_readings_vm_cpu_readings-file-1-of-195.csv.gz"
)
READINGS_FILE = os.path.join(RAW_DIR, "vm_cpu_readings-file-1-of-195.csv.gz")



def download_file(url, dest):
    if os.path.exists(dest):
        print(f"[SKIP] Already downloaded: {dest}")
        return
    print(f"[DOWNLOAD] {url}")
    print("This may take 10-15 minutes depending on your internet speed...")
    urllib.request.urlretrieve(url, dest)
    print(f"[DONE] Saved to {dest}")



print("=" * 60)
print("STEP 1: DOWNLOADING AZURE VM DATASET")
print("=" * 60)
download_file(READINGS_URL, READINGS_FILE)


print("\n" + "=" * 60)
print("STEP 2: LOADING & CLEANING")
print("=" * 60)


readings_cols = ['timestamp', 'vm_id', 'min_cpu', 'max_cpu', 'avg_cpu']
chunks = []
for i, chunk in enumerate(pd.read_csv(
    READINGS_FILE,
    compression='gzip',
    header=None,
    names=readings_cols,
    chunksize=500_000,
    low_memory=False
)):
    chunks.append(chunk)
    if (i + 1) % 5 == 0:

        print(f"   Processed {(i+1)*500_000:,} rows...")


df = pd.concat(chunks, ignore_index=True)
print(f"Total loaded: {len(df):,} rows, {df['vm_id'].nunique():,} VMs")


# Clean
df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp'])
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', origin='unix')
df = df.sort_values(['vm_id', 'timestamp'])


for col in ['min_cpu', 'max_cpu', 'avg_cpu']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[df[col].between(0, 100)]


df = df[df['max_cpu'] >= df['min_cpu']]


# Filter to VMs with at least 20 readings
vm_counts = df.groupby('vm_id').size()
valid_vms = vm_counts[vm_counts >= 20].index
df = df[df['vm_id'].isin(valid_vms)]


print(f"After cleaning: {len(df):,} rows, {df['vm_id'].nunique():,} VMs")
time_range_hours = (df['timestamp'].max() - df['timestamp'].min()) / 3600
print(f"Time range: {time_range_hours:.1f} hours")


# Save cleaned data
df.to_parquet(os.path.join(CLEAN_DIR, "azure_vm_clean.parquet"), index=False)
print(f"[SAVED] {CLEAN_DIR}/azure_vm_clean.parquet")


# Save aggregate cluster demand (for ARIMA): mean CPU % across VMs, so its
# RMSE is in the same units (CPU %) as the per-VM models
cluster_demand = (
    df.groupby('timestamp')['avg_cpu']
    .mean()
    .reset_index()
    .sort_values('timestamp')
)
cluster_demand.to_parquet(os.path.join(CLEAN_DIR, "cluster_demand.parquet"), index=False)
print(f"[SAVED] {CLEAN_DIR}/cluster_demand.parquet (aggregate)")
print("\n[DONE] Preprocessing complete!")
