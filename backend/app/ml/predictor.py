"""
Unified inference engine for Kochi Metro predictive maintenance.
Serves t+3 and t+7 forecasts using cached models.
"""
from typing import Union, Dict, Any, List, Optional
import numpy as np
import pandas as pd
from .model_loader import ModelLoader
from .feature_pipeline import FeaturePipeline
from ..config import FEATURE_COLUMNS

class MaintenancePredictor:
    def __init__(self, loader: Optional[ModelLoader] = None):
        self.loader = loader or ModelLoader()

    def predict(
        self,
        features: Union[pd.DataFrame, np.ndarray, List[Dict[str, Any]]],
        horizon: str = "t3",
        model_type: str = "ensemble"
    ) -> np.ndarray:
        """
        Generates priority score forecasts for day t+3 or t+7.

        Args:
            features: 24-feature DataFrame, NumPy matrix, or list of dicts.
            horizon: 't3' (short-term dispatch) or 't7' (weekly overhaul).
            model_type: 'ensemble' (Dual-Booster SLSQP), 'xgboost', or 'lightgbm'.

        Returns:
            np.ndarray of predicted maintenance priority scores in [0.001, 0.999].
        """
        # Prepare feature DataFrame
        if isinstance(features, np.ndarray):
            if features.shape[1] != len(FEATURE_COLUMNS):
                raise ValueError(f"Expected {len(FEATURE_COLUMNS)} columns, got {features.shape[1]}")
            df_features = pd.DataFrame(features, columns=FEATURE_COLUMNS)
        else:
            df_features = FeaturePipeline.extract_features(features)

        model = self.loader.get_model(horizon=horizon, model_type=model_type)

        if isinstance(model, dict) and 'weights' in model:
            # Dual-Booster SLSQP Ensemble Bundle
            base_models = model['base_models']
            weights = model['weights']
            
            preds = []
            for m_name in model['model_names']:
                m = base_models[m_name]
                cols = list(m.feature_names_in_) if hasattr(m, 'feature_names_in_') else FEATURE_COLUMNS
                preds.append(m.predict(df_features[cols]))
                
            P = np.column_stack(preds)
            w = np.array([weights[m_name] for m_name in model['model_names']])
            final_pred = P @ w
        else:
            # Standalone Regressor
            cols = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else FEATURE_COLUMNS
            final_pred = model.predict(df_features[cols])

        # Clip and round to maintain statutory score bounds
        return np.round(np.clip(final_pred, 0.001, 0.999), 4)

