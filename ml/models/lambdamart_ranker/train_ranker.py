#!/usr/bin/env python3
"""
LambdaMART Ranker for Kochi Metro Rolling Stock Maintenance Priority Forecasting.
"""
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import lightgbm as lgb
from ml.common.data import load_data, get_chronological_splits, FEATURE_COLUMNS
from ml.common.metrics import evaluate_metrics
from ml.common.runner import save_model_results, parse_horizon_arg

os.environ['MPLCONFIGDIR'] = '/tmp/mpl'

OUTPUT_DIR = Path(__file__).resolve().parent

PARAMS = {
    'objective': 'lambdarank',
    'learning_rate': 0.05,
    'n_estimators': 200,
    'num_leaves': 31,
    'min_child_samples': 10,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1
}

def train_ranker(horizon: int = 3):
    df, target_col = load_data(horizon=horizon)
    train_df, val_df, test_df = get_chronological_splits(df)
    
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    q_train_val = train_val_df.groupby('date', sort=False).size().values
    
    final_ranker = lgb.LGBMRanker(**PARAMS)
    final_ranker.fit(train_val_df[FEATURE_COLUMNS], train_val_df['relevance_grade'], group=q_train_val)
    
    # Predict raw ranking scores & min-max calibrate for regression metrics
    raw_test_pred = final_ranker.predict(test_df[FEATURE_COLUMNS])
    t_min, t_max = df[target_col].min(), df[target_col].max()
    p_min, p_max = raw_test_pred.min(), raw_test_pred.max()
    calibrated_test_pred = t_min + (raw_test_pred - p_min) / (p_max - p_min + 1e-9) * (t_max - t_min)
    
    metrics = evaluate_metrics(test_df[target_col].values, calibrated_test_pred, test_df, target_col)
    
    save_model_results(
        output_dir=OUTPUT_DIR,
        model_name="LambdaMART Ranker",
        horizon=horizon,
        metrics=metrics,
        model=final_ranker,
        primary_filename=f"lambdamart_ranker_t{horizon}.joblib",
        aliases=[f"ranker_model_t{horizon}.joblib"],
        hyperparameters=PARAMS
    )

def main():
    horizons = parse_horizon_arg([3, 7])
    for h in horizons:
        train_ranker(horizon=h)

if __name__ == '__main__':
    main()
