# Quick check: real feature importance from YOUR saved XGBoost regressor.
# Run this once before recording so you can cite verified numbers, not
# the group paper's figures (train_models.py never actually logs
# feature importance anywhere, so that 37.7% figure isn't reproducible
# from this codebase as written).
#
# Run from your project root: python check_feature_importance.py
import xgboost as xgb

model = xgb.Booster()
model.load_model("models/xgb_regressor.json")

importance = model.get_score(importance_type='gain')
total = sum(importance.values())
ranked = sorted(importance.items(), key=lambda x: -x[1])

print("Top 5 features by importance (gain):")
for name, score in ranked[:5]:
    print(f"  {name}: {score/total*100:.1f}%")
