#!/usr/bin/env python3
"""
Method 2: Regularized Stacking Meta-Learner (RidgeCV) Ensemble
(XGBoost + LightGBM + HistGradientBoosting)
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
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
from ml.common.data import load_data, get_chronological_splits, FEATURE_COLUMNS
from ml.common.metrics import evaluate_metrics
from ml.common.runner import parse_horizon_arg

os.environ['MPLCONFIGDIR'] = '/tmp/mpl'
OUTPUT_DIR = Path(__file__).resolve().parent

def get_validation_predictions(train_df, val_df, target_col):
    """Trains leak-free base models on Train split to evaluate on Validation split."""
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[target_col]
    X_val = val_df[FEATURE_COLUMNS]
    
    hgb = HistGradientBoostingRegressor(
        learning_rate=0.1, max_iter=150, max_leaf_nodes=15,
        min_samples_leaf=20, l2_regularization=1.0, random_state=42
    )
    hgb.fit(X_train, y_train)
    
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
        'HistGradientBoosting': hgb.predict(X_val),
        'XGBoost': xgb_mod.predict(X_val),
        'LightGBM': lgbm.predict(X_val)
    }

def train_and_eval(horizon: int = 3):
    print(f"\n==================== METHOD 2 (STACKING RIDGECV) FOR t+{horizon} ====================")
    df, target_col = load_data(horizon=horizon)
    train_df, val_df, test_df = get_chronological_splits(df)
    
    base_model_paths = {
        'HistGradientBoosting': f'ml/models/hist_gradient_boosting/hist_gradient_boosting_t{horizon}.joblib',
        'XGBoost': f'ml/models/xgboost/xgboost_t{horizon}.joblib',
        'LightGBM': f'ml/models/lightgbm/lightgbm_t{horizon}.joblib'
    }
    model_names = list(base_model_paths.keys())
    base_models = {m: joblib.load(base_model_paths[m]) for m in model_names}
    
    # 1. Validation predictions & Stacking RidgeCV
    val_preds = get_validation_predictions(train_df, val_df, target_col)
    P_val = np.column_stack([val_preds[m] for m in model_names])
    y_val = val_df[target_col].values
    
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0]
    meta_model = RidgeCV(alphas=alphas, cv=5)
    meta_model.fit(P_val, y_val)
    
    coef_dict = {
        'intercept': round(float(meta_model.intercept_), 5),
        'alpha': round(float(meta_model.alpha_), 4),
        'coefficients': {model_names[i]: round(float(meta_model.coef_[i]), 4) for i in range(len(model_names))}
    }
    print(f"Optimal Stacking Meta-Learner for t+{horizon}: Alpha = {meta_model.alpha_}, Coefs = {coef_dict['coefficients']}")
    
    # 2. Test predictions & evaluation
    P_test = np.column_stack([
        base_models[m].predict(test_df[list(base_models[m].feature_names_in_)])
        for m in model_names
    ])
    final_test_pred = meta_model.predict(P_test)
    metrics = evaluate_metrics(test_df[target_col].values, final_test_pred, test_df, target_col)
    
    # 3. Save artifacts
    model_path = OUTPUT_DIR / f"ensemble_stacking_t{horizon}.joblib"
    joblib.dump({
        'method': 'Regularized Stacking Meta-Learner (RidgeCV)',
        'horizon': f't+{horizon}',
        'model_names': model_names,
        'meta_model': meta_model,
        'coefficients': coef_dict,
        'base_models': base_models
    }, model_path)
    
    coef_path = OUTPUT_DIR / f"coefficients_t{horizon}.json"
    with open(coef_path, 'w') as f:
        json.dump(coef_dict, f, indent=2)
        
    json_path = OUTPUT_DIR / f"result_t{horizon}.json"
    with open(json_path, 'w') as f:
        json.dump({
            "ensemble_method": "Method 2: Regularized Stacking Meta-Learner (RidgeCV)",
            "horizon": f"t+{horizon}",
            "meta_model_details": coef_dict,
            "metrics": metrics
        }, f, indent=2)
        
    print(f"Saved: {model_path}, {coef_path}, and {json_path}")
    print(json.dumps(metrics, indent=2))

def main():
    horizons = parse_horizon_arg([3, 7])
    for h in horizons:
        train_and_eval(horizon=h)

if __name__ == '__main__':
    main()
