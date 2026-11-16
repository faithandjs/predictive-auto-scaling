# Predictive Auto-Scaling

Forecasts each virtual machine's CPU usage **15 minutes ahead** from real Microsoft Azure traces, and scales up *before* a VM overloads instead of after. A cost simulation compares this with a standard reactive rule: scale up when CPU is above 75% right now.

Built for COM747 (Data Science & Machine Learning).

## Results

Test set: the last 30% of the trace by time, 49,880 readings from 4,968 VMs.

**Cost simulation.** An unneeded scale-up costs $0.10 and a missed overload costs $1.00.

| Policy | Overloads caught | Unneeded scale-ups | Missed overloads | Cost |
|---|---|---|---|---|
| Reactive (CPU > 75% now) | 650 | 107 | 113 | $123.70 |
| Predictive (XGBoost, 15 min ahead) | 675 | 145 | 88 | $102.50 |

The predictive policy costs **17.1% less** and misses **22.1% fewer** overloads.

**Models**

| Model | Predicts | RMSE (CPU %) | F1 | AUC |
|---|---|---|---|---|
| XGBoost | per-VM CPU in 15 min, plus overload yes/no | 5.54 | 0.853 | 0.983 |
| LSTM (tuned with Optuna) | per-VM CPU in 15 min | 6.53 | 0.729 | 0.978 |
| ARIMA(2,1,2) | cluster-average CPU | 0.13* | n/a | n/a |

\*ARIMA forecasts one smooth cluster-wide average, so its RMSE isn't comparable to the per-VM models.

XGBoost walk-forward validation (3 folds, split by time): RMSE 5.63 ± 0.16.

Open [`dashboard.html`](dashboard.html) in a browser to see all 10 charts.

## How it works

`run_all.sh` runs five steps in order:

1. **`preprocess.py`** downloads file 1 of 195 of the [Azure Public Dataset v2](https://github.com/Azure/AzurePublicDataset) VM CPU readings (about 800 MB, one reading per VM every 5 minutes). It removes invalid rows and VMs with fewer than 20 readings.
2. **`features.py`** samples 5,000 VMs and builds 19 inputs per reading:
   - lagged CPU values (1, 2, 3 and 5 steps back)
   - rolling means and standard deviations over the previous 3 and 5 readings
   - recent change in CPU
   - time of day

   The targets are CPU 3 steps (15 minutes) ahead, and whether that value is above 75%. The data is split by time (70/30), and inputs are standardised using the training data only.
3. **`train_models.py`** trains ARIMA, XGBoost (a regressor and a classifier) and an LSTM.
   - XGBoost is checked with walk-forward validation.
   - The LSTM is tuned with Optuna using a validation split taken from the training data, so the test set is used only for the final score.
   - A simple drift detector flags prediction errors more than 3 standard deviations above the recent average.
4. **`simulate.py`** runs the reactive vs predictive cost simulation and writes the chart data to `visualisations/`.
5. **`build_dashboard.py`** builds the standalone `dashboard.html` (Chart.js).

## Running it

Requires Python 3.11+ and about 8 GB of RAM.

```bash
python -m venv venv && source venv/bin/activate
pip install pandas numpy scikit-learn statsmodels xgboost torch optuna fastapi "uvicorn[standard]" pyarrow
bash run_all.sh
```

The first run downloads the dataset, which can take 10–15 minutes. The Optuna tuning step is the slowest part of training.

Or with Docker:

```bash
docker build -t predictive-auto-scaling .
docker run predictive-auto-scaling
```

`data/` is not committed: it is about 1 GB and is rebuilt by steps 1–2. The trained models, results and chart data from the last run are committed.

## Inference API

`inference_api.py` serves the trained XGBoost models with FastAPI. Run steps 1–3 first: the API also needs the scaler saved in `data/features/scaler.pkl`.

```bash
python inference_api.py   # http://localhost:8000, docs at /docs
```

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"vm_id": "vm-1", "timestamp": 1500, "avg_cpu": 62.0, "max_cpu": 80.0, "min_cpu": 40.0}'
```

The API keeps the last 50 readings for each VM. Once it has at least 6 readings for a VM, it predicts CPU 15 minutes ahead and recommends scaling up if the prediction is above 70% (the 75% threshold minus a 5-point safety buffer). Until then, it falls back to the reactive rule. Predictions take about 2 ms.

Other endpoints: `GET /health` and `GET /stats`.

## Project structure

```
preprocess.py               Step 1: download and clean the Azure traces
features.py                 Step 2: feature engineering and train/test split
train_models.py             Step 3: ARIMA, XGBoost and LSTM training and evaluation
simulate.py                 Step 4: cost simulation and chart data
build_dashboard.py          Step 5: generate dashboard.html
inference_api.py            FastAPI service for live predictions
check_feature_importance.py Prints the top XGBoost inputs by gain
run_all.sh, Dockerfile      Run the full pipeline
models/                     Trained XGBoost (JSON) and LSTM (PyTorch) models
results/                    Metrics, predictions, simulation summary
visualisations/             Data behind the dashboard charts
```

## Limitations

- **About 3.7 hours of data.** Only 1 of the 195 trace files is used, to keep memory use manageable. The time-of-day input and the error-by-time-of-day chart cover only a small part of a day, and daily patterns can't be learned.
- **Overloads are rare** (about 1.5% of test readings). Results depend on the 75% threshold and on the assumed 10:1 cost of a missed overload versus an unneeded scale-up. `visualisations/threshold_sensitivity.json` shows how cost changes with the decision threshold.
- **The cost model is simplified.** It counts decisions only and ignores how long scaling takes to start and how much capacity is added.
- **The API stores VM history in memory**, so it is lost on restart and not shared between server processes.
