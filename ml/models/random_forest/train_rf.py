#!/usr/bin/env python3
"""
Random Forest Regressor for Kochi Metro Rolling Stock Maintenance Priority Forecasting.
"""
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sklearn.ensemble import RandomForestRegressor
from ml.common.runner import train_standard_regressor, parse_horizon_arg

os.environ['MPLCONFIGDIR'] = '/tmp/mpl'

# Model Hyperparameters
PARAMS = {
    'n_estimators': 200,
    'max_depth': 16,
    'min_samples_split': 2,
    'min_samples_leaf': 2,
    'max_features': 0.8,
    'random_state': 42,
    'n_jobs': -1
}

OUTPUT_DIR = Path(__file__).resolve().parent

def main():
    horizons = parse_horizon_arg([3, 7])
    for h in horizons:
        train_standard_regressor(
            model_factory=lambda **p: RandomForestRegressor(**p),
            model_name="Random Forest Regressor",
            output_dir=OUTPUT_DIR,
            file_prefix="random_forest",
            params=PARAMS,
            horizon=h,
            aliases=["rf_model"]
        )

if __name__ == '__main__':
    main()
