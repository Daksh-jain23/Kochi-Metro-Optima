#!/usr/bin/env python3
"""
Ridge Regression for Kochi Metro Rolling Stock Maintenance Priority Forecasting.
"""
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from ml.common.data import load_data, get_chronological_splits, FEATURE_COLUMNS
from ml.common.metrics import evaluate_metrics
from ml.common.runner import save_model_results, parse_horizon_arg

os.environ['MPLCONFIGDIR'] = '/tmp/mpl'

OUTPUT_DIR = Path(__file__).resolve().parent
ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]

def train_ridge(horizon: int = 3):
    print(f"\n==================== TRAINING RIDGE REGRESSION FOR t+{horizon} ====================")
    df, target_col = load_data(horizon=horizon)
    train_df, val_df, test_df = get_chronological_splits(df)
    
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[target_col]
    X_val, y_val = val_df[FEATURE_COLUMNS], val_df[target_col]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[target_col]
    
    # Validation Alpha Tuning
    best_alpha, best_val_mae = None, float('inf')
    for a in ALPHAS:
        pipe = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=a, random_state=42))])
        pipe.fit(X_train, y_train)
        v_mae = mean_absolute_error(y_val, pipe.predict(X_val))
        if v_mae < best_val_mae:
            best_val_mae = v_mae
            best_alpha = a
            
    print(f"Optimal Alpha for t+{horizon}: {best_alpha} (Val MAE: {best_val_mae:.5f})")
    
    # Train on Train + Val
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    final_model = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=best_alpha, random_state=42))])
    final_model.fit(train_val_df[FEATURE_COLUMNS], train_val_df[target_col])
    
    # Evaluate on Test
    test_pred = final_model.predict(X_test)
    metrics = evaluate_metrics(y_test, test_pred, test_df, target_col)
    
    save_model_results(
        output_dir=OUTPUT_DIR,
        model_name="Ridge Regression",
        horizon=horizon,
        metrics=metrics,
        model=final_model,
        primary_filename=f"ridge_model_t{horizon}.joblib",
        aliases=[f"ridge_t{horizon}.joblib"],
        hyperparameters={"alpha": best_alpha}
    )

def main():
    horizons = parse_horizon_arg([3, 7])
    for h in horizons:
        train_ridge(horizon=h)

if __name__ == '__main__':
    main()
