#!/bin/bash
set -e  # stop immediately if any step fails

echo "=================================================="
echo "STEP 1/5: preprocess.py"
echo "=================================================="
python preprocess.py

echo "=================================================="
echo "STEP 2/5: features.py"
echo "=================================================="
python features.py

echo "=================================================="
echo "STEP 3/5: train_models.py"
echo "=================================================="
python train_models.py

echo "=================================================="
echo "STEP 4/5: simulate.py"
echo "=================================================="
python simulate.py

echo "=================================================="
echo "STEP 5/5: build_dashboard.py"
echo "=================================================="
python build_dashboard.py

echo "=================================================="
echo "PIPELINE COMPLETE — check dashboard.html"
echo "=================================================="
