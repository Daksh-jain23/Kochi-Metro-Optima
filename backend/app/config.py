"""
Backend configuration constants, model artifact paths, and feature definitions.
"""
from pathlib import Path
from typing import List

# Base Paths
BACKEND_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BACKEND_DIR / "models"
METADATA_PATH = MODELS_DIR / "metadata.json"

# Production Model Artifact Paths
MODEL_PATHS = {
    "t3": {
        "ensemble": MODELS_DIR / "t3" / "ensemble_slsqp_xgb_lgbm_t3.joblib",
        "xgboost": MODELS_DIR / "t3" / "xgboost_t3.joblib",
        "lightgbm": MODELS_DIR / "t3" / "lightgbm_t3.joblib",
    },
    "t7": {
        "ensemble": MODELS_DIR / "t7" / "ensemble_slsqp_xgb_lgbm_t7.joblib",
        "xgboost": MODELS_DIR / "t7" / "xgboost_t7.joblib",
        "lightgbm": MODELS_DIR / "t7" / "lightgbm_t7.joblib",
    }
}

# 24 Canonical Model Features
FEATURE_COLUMNS: List[str] = [
    'days_remaining', 'critical_jobs', 'non_critical_jobs', 'revenue',
    'remaining_hours', 'penalty', 'total_km', 'daily_avg_run',
    'variance', 'mileage', 'train_age_years', 'daily_trips',
    'days_since_last_maintenance', 'maintenance_priority_score',
    'rolling_7d_critical_jobs', 'rolling_7d_non_critical_jobs',
    'rolling_7d_daily_avg_run', 'rolling_7d_revenue', 'rolling_7d_penalty',
    'delta_7d_critical_jobs', 'delta_7d_days_remaining', 'delta_7d_total_km',
    'delta_7d_priority_score', 'delta_7d_days_since_maint'
]

# 14 Raw Observable Features (Prior to rolling window computation)
RAW_FEATURE_COLUMNS: List[str] = [
    'days_remaining', 'critical_jobs', 'non_critical_jobs', 'revenue',
    'remaining_hours', 'penalty', 'total_km', 'daily_avg_run',
    'variance', 'mileage', 'train_age_years', 'daily_trips',
    'days_since_last_maintenance', 'maintenance_priority_score'
]

# Dispatch Categorization Rank Thresholds
RANK_CUTOFF_CRITICAL_DEPOT = 3
RANK_CUTOFF_INSPECTION_BAY = 5

LABEL_CRITICAL_DEPOT = "CRITICAL: Depot"
LABEL_INSPECTION_BAY = "INSPECTION: Bay"
LABEL_ACTIVE_SERVICE = "ACTIVE SERVICE"

