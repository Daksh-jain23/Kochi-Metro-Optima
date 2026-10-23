#!/usr/bin/env python3
"""
HistGradientBoosting Regressor for Kochi Metro Rolling Stock Maintenance Priority Forecasting.
"""
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sklearn.ensemble import HistGradientBoostingRegressor
from ml.common.runner import train_standard_regressor, parse_horizon_arg

os.environ['MPLCONFIGDIR'] = '/tmp/mpl'

# Model Hyperparameters
PARAMS = {
    'learning_rate': 0.1,
    'max_iter': 150,
    'max_leaf_nodes': 15,
    'min_samples_leaf': 20,
    'l2_regularization': 1.0,
    'random_state': 42
}

OUTPUT_DIR = Path(__file__).resolve().parent

def main():
    horizons = parse_horizon_arg([3, 7])
    for h in horizons:
        train_standard_regressor(
            model_factory=lambda **p: HistGradientBoostingRegressor(**p),
            model_name="HistGradientBoosting Regressor",
            output_dir=OUTPUT_DIR,
            file_prefix="hist_gradient_boosting",
            params=PARAMS,
            horizon=h,
            aliases=["hgb_model"]
        )

if __name__ == '__main__':
    main()
