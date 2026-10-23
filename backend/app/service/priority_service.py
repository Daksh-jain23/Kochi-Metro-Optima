"""
High-level service orchestrating fleet maintenance priority forecasting,
ranking, and depot dispatch scheduling recommendations.
"""
from typing import List, Dict, Any, Union, Optional
import numpy as np
import pandas as pd
from ..ml.predictor import MaintenancePredictor
from ..ml.feature_pipeline import FeaturePipeline
from ..schemas.prediction import TrainPriorityPrediction, FleetPriorityResponse
from ..config import (
    RANK_CUTOFF_CRITICAL_DEPOT,
    RANK_CUTOFF_INSPECTION_BAY,
    LABEL_CRITICAL_DEPOT,
    LABEL_INSPECTION_BAY,
    LABEL_ACTIVE_SERVICE,
    FEATURE_COLUMNS,
)

class PriorityService:
    def __init__(self, predictor: Optional[MaintenancePredictor] = None):
        self.predictor = predictor or MaintenancePredictor()

    def rank_fleet(
        self,
        fleet_data: Union[pd.DataFrame, List[Dict[str, Any]]],
        horizon: str = "t3",
        model_type: str = "ensemble",
        evaluation_date: Optional[str] = None
    ) -> FleetPriorityResponse:
        """
        Forecasts future priority scores and produces ranked depot dispatch recommendations.

        Args:
            fleet_data: DataFrame or list of dicts with train operational telemetry.
            horizon: 't3' (short-term dispatch) or 't7' (weekly overhaul).
            model_type: 'ensemble' (Dual-Booster SLSQP), 'xgboost', or 'lightgbm'.
            evaluation_date: Optional string identifier for current day.
        """
        if isinstance(fleet_data, list):
            df = pd.DataFrame(fleet_data)
        else:
            df = fleet_data.copy()

        train_ids = df['train_id'].tolist() if 'train_id' in df.columns else [f"TRAIN_{i+1:02d}" for i in range(len(df))]
        
        # Calculate or extract current-day score
        if 'maintenance_priority_score' in df.columns:
            current_scores = df['maintenance_priority_score'].values
        else:
            current_scores = FeaturePipeline.calculate_priority_score(df)

        # Generate feature matrix & run predictions
        df_features = FeaturePipeline.extract_features(df)
        forecast_scores = self.predictor.predict(df_features, horizon=horizon, model_type=model_type)

        # Assemble ranking DataFrame
        ranked_df = pd.DataFrame({
            'train_id': train_ids,
            'forecast_score': forecast_scores,
            'current_score': current_scores
        }).sort_values('forecast_score', ascending=False).reset_index(drop=True)

        ranked_df['rank'] = np.arange(1, len(ranked_df) + 1)

        def assign_status(rank: int) -> str:
            if rank <= RANK_CUTOFF_CRITICAL_DEPOT:
                return LABEL_CRITICAL_DEPOT
            elif rank <= RANK_CUTOFF_INSPECTION_BAY:
                return LABEL_INSPECTION_BAY
            return LABEL_ACTIVE_SERVICE

        ranked_df['status'] = ranked_df['rank'].apply(assign_status)

        predictions: List[TrainPriorityPrediction] = []
        for _, row in ranked_df.iterrows():
            pred = TrainPriorityPrediction(
                train_id=str(row['train_id']),
                forecast_score=float(row['forecast_score']),
                current_score=float(row['current_score']),
                rank=int(row['rank']),
                status=str(row['status']),
                horizon=f"{horizon.upper()}",
                details={
                    "delta_vs_current": round(float(row['forecast_score'] - row['current_score']), 4)
                }
            )
            predictions.append(pred)

        crit_count = sum(1 for p in predictions if p.status == LABEL_CRITICAL_DEPOT)
        insp_count = sum(1 for p in predictions if p.status == LABEL_INSPECTION_BAY)
        act_count = sum(1 for p in predictions if p.status == LABEL_ACTIVE_SERVICE)

        model_display = "Dual-Booster SLSQP (XGBoost + LightGBM)" if model_type == "ensemble" else model_type.upper()

        return FleetPriorityResponse(
            horizon=f"{horizon.upper()}",
            model_used=model_display,
            evaluation_date=evaluation_date,
            total_trains=len(predictions),
            critical_depot_count=crit_count,
            inspection_bay_count=insp_count,
            active_service_count=act_count,
            predictions=predictions
        )

