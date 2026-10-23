#!/usr/bin/env python3
"""
LightGBM Regressor for Kochi Metro Rolling Stock Maintenance Priority Forecasting.
"""
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import lightgbm as lgb
from ml.common.runner import train_standard_regressor, parse_horizon_arg

os.environ['MPLCONFIGDIR'] = '/tmp/mpl'

# Model Hyperparameters
PARAMS = {
    'learning_rate': 0.03,
    'n_estimators': 200,
    'num_leaves': 31,
    'min_child_samples': 10,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

OUTPUT_DIR = Path(__file__).resolve().parent

def main():
    horizons = parse_horizon_arg([3, 7])
    for h in horizons:
        train_standard_regressor(
            model_factory=lambda **p: lgb.LGBMRegressor(**p),
            model_name="LightGBM Regressor",
            output_dir=OUTPUT_DIR,
            file_prefix="lightgbm",
            params=PARAMS,
            horizon=h,
            aliases=["lgbm_model"]
        )

if __name__ == '__main__':
    main()
