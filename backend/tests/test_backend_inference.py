"""
Comprehensive backend inference and integration test suite.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from backend.app.ml.model_loader import ModelLoader
from backend.app.ml.feature_pipeline import FeaturePipeline
from backend.app.ml.predictor import MaintenancePredictor
from backend.app.service.priority_service import PriorityService
from backend.app.config import FEATURE_COLUMNS

def test_model_loader_ensemble_t3():
    model = ModelLoader.get_model(horizon="t3", model_type="ensemble")
    assert isinstance(model, dict)
    assert "weights" in model
    assert "base_models" in model
    assert "XGBoost" in model["weights"]
    assert "LightGBM" in model["weights"]
    assert model["weights"]["XGBoost"] == 0.5
    assert model["weights"]["LightGBM"] == 0.5

def test_model_loader_ensemble_t7():
    model = ModelLoader.get_model(horizon="t7", model_type="ensemble")
    assert isinstance(model, dict)
    assert "weights" in model
    assert model["weights"]["XGBoost"] == 0.5
    assert model["weights"]["LightGBM"] == 0.5

def test_model_loader_standalone():
    xgb = ModelLoader.get_model(horizon="t7", model_type="xgboost")
    assert hasattr(xgb, "predict")
    lgb = ModelLoader.get_model(horizon="t3", model_type="lightgbm")
    assert hasattr(lgb, "predict")

def test_priority_service_fleet_ranking():
    # Load sample operational day from data_2026
    df = pd.read_csv("ml/data/data_2026/railway_maintenance_365_days.csv")
    sample_day = df[df['date'] == '2026-06-15'].copy()
    assert len(sample_day) == 25

    service = PriorityService()

    # Test t+3 Dual-Booster SLSQP Ensemble
    resp_t3 = service.rank_fleet(sample_day, horizon="t3", model_type="ensemble", evaluation_date="2026-06-15")
    assert resp_t3.total_trains == 25
    assert resp_t3.critical_depot_count == 3
    assert resp_t3.inspection_bay_count == 2
    assert resp_t3.active_service_count == 20
    assert len(resp_t3.predictions) == 25

    # Check descending sort
    scores = [p.forecast_score for p in resp_t3.predictions]
    assert scores == sorted(scores, reverse=True)

    # Check status assignment
    assert resp_t3.predictions[0].status == "CRITICAL: Depot"
    assert resp_t3.predictions[1].status == "CRITICAL: Depot"
    assert resp_t3.predictions[2].status == "CRITICAL: Depot"
    assert resp_t3.predictions[3].status == "INSPECTION: Bay"
    assert resp_t3.predictions[4].status == "INSPECTION: Bay"
    assert resp_t3.predictions[5].status == "ACTIVE SERVICE"
    assert resp_t3.predictions[24].status == "ACTIVE SERVICE"

    # Test t+7 Dual-Booster SLSQP Ensemble
    resp_t7 = service.rank_fleet(sample_day, horizon="t7", model_type="ensemble", evaluation_date="2026-06-15")
    assert resp_t7.total_trains == 25
    assert resp_t7.critical_depot_count == 3
    assert resp_t7.inspection_bay_count == 2
    assert resp_t7.active_service_count == 20

    print("\n[SUCCESS] Backend PriorityService passed all integration tests!")

if __name__ == "__main__":
    test_model_loader_ensemble_t3()
    test_model_loader_ensemble_t7()
    test_model_loader_standalone()
    test_priority_service_fleet_ranking()
    print("All backend tests completed successfully!")
