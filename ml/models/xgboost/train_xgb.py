#!/usr/bin/env python3
"""
XGBoost Regressor for Kochi Metro Rolling Stock Maintenance Priority Forecasting.
"""
import os
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import xgboost as xgb
from ml.common.runner import train_standard_regressor, parse_horizon_arg

os.environ['MPLCONFIGDIR'] = '/tmp/mpl'

# Model Hyperparameters
PARAMS = {
    'learning_rate': 0.05,
    'n_estimators': 150,
    'max_depth': 5,
    'subsample': 0.85,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1
}

OUTPUT_DIR = Path(__file__).resolve().parent

def main():
    horizons = parse_horizon_arg([3, 7])
    for h in horizons:
        train_standard_regressor(
            model_factory=lambda **p: xgb.XGBRegressor(**p),
            model_name="XGBoost Regressor",
            output_dir=OUTPUT_DIR,
            file_prefix="xgboost",
            params=PARAMS,
            horizon=h,
            aliases=["xgb_model"]
        )

if __name__ == '__main__':
    main()
