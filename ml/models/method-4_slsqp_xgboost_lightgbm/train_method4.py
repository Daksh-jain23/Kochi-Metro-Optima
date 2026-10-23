#!/usr/bin/env python3
"""
Method 4: Constrained Convex Optimization (SLSQP) Ensemble (XGBoost + LightGBM)
Kochi Metro Rolling Stock Maintenance Priority Forecasting.
"""
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import joblib
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error
from ml.common.data import load_data, get_chronological_splits, FEATURE_COLUMNS
from ml.common.metrics import evaluate_metrics
from ml.common.runner import parse_horizon_arg

os.environ['MPLCONFIGDIR'] = '/tmp/mpl'
OUTPUT_DIR = Path(__file__).resolve().parent

def get_validation_predictions(train_df, val_df, target_col):
    """Trains leak-free base models on Train split to evaluate on Validation split."""
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[target_col]
    X_val = val_df[FEATURE_COLUMNS]
    
    xgb_mod = xgb.XGBRegressor(
        learning_rate=0.05, n_estimators=150, max_depth=5,
        subsample=0.85, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1
    )
    xgb_mod.fit(X_train, y_train)
    
    lgbm = lgb.LGBMRegressor(
        learning_rate=0.03, n_estimators=200, num_leaves=31,
        min_child_samples=10, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbose=-1
    )
    lgbm.fit(X_train, y_train)
    
    return {
        'XGBoost': xgb_mod.predict(X_val),
        'LightGBM': lgbm.predict(X_val)
    }

def train_and_eval(horizon: int = 3):
    print(f"\n==================== METHOD 4 (SLSQP XGBOOST + LIGHTGBM) FOR t+{horizon} ====================")
    df, target_col = load_data(horizon=horizon)
    train_df, val_df, test_df = get_chronological_splits(df)
    
    base_model_paths = {
        'XGBoost': f'ml/models/xgboost/xgboost_t{horizon}.joblib',
        'LightGBM': f'ml/models/lightgbm/lightgbm_t{horizon}.joblib'
    }
    model_names = list(base_model_paths.keys())
    base_models = {m: joblib.load(base_model_paths[m]) for m in model_names}
    
    # 1. Validation predictions & SLSQP optimization
    val_preds = get_validation_predictions(train_df, val_df, target_col)
    P_val = np.column_stack([val_preds[m] for m in model_names])
    y_val = val_df[target_col].values
    
    def objective(w):
        return mean_absolute_error(y_val, P_val @ w)
        
    init_w = np.ones(len(model_names)) / len(model_names)
    bounds = [(0.0, 1.0) for _ in model_names]
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    
    res = minimize(objective, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
    weights = np.maximum(0, res.x)
    weights /= np.sum(weights)
    weights_dict = {model_names[i]: round(float(weights[i]), 4) for i in range(len(model_names))}
    print(f"Optimal SLSQP Weights for t+{horizon}: {weights_dict}")
    
    # 2. Test predictions & evaluation
    P_test = np.column_stack([
        base_models[m].predict(test_df[list(base_models[m].feature_names_in_)])
        for m in model_names
    ])
    final_test_pred = P_test @ weights
    metrics = evaluate_metrics(test_df[target_col].values, final_test_pred, test_df, target_col)
    
    # 3. Save artifacts
    model_path = OUTPUT_DIR / f"ensemble_slsqp_xgb_lgbm_t{horizon}.joblib"
    joblib.dump({
        'method': 'Constrained Convex Optimization (SLSQP) XGBoost + LightGBM',
        'horizon': f't+{horizon}',
        'model_names': model_names,
        'weights': weights_dict,
        'base_models': base_models
    }, model_path)
    
    weights_path = OUTPUT_DIR / f"weights_t{horizon}.json"
    with open(weights_path, 'w') as f:
        json.dump(weights_dict, f, indent=2)
        
    json_path = OUTPUT_DIR / f"result_t{horizon}.json"
    with open(json_path, 'w') as f:
        json.dump({
            "ensemble_method": "Method 4: Constrained Convex Optimization (SLSQP) XGBoost + LightGBM",
            "horizon": f"t+{horizon}",
            "weights": weights_dict,
            "metrics": metrics
        }, f, indent=2)
        
    print(f"Saved: {model_path}, {weights_path}, and {json_path}")
    print(json.dumps(metrics, indent=2))

def main():
    horizons = parse_horizon_arg([3, 7])
    for h in horizons:
        train_and_eval(horizon=h)

if __name__ == '__main__':
    main()
