"""
Feature engineering and validation pipeline for Kochi Metro maintenance forecasting.
"""
import numpy as np
import pandas as pd
from typing import List, Union, Dict, Any
from ..config import FEATURE_COLUMNS

class FeaturePipeline:
    """Transforms raw train records into model-ready 24-dimensional feature arrays."""

    @staticmethod
    def calculate_priority_score(df: pd.DataFrame) -> np.ndarray:
        """
        Calculates the exact 4-tier ground-truth maintenance priority score:
        Safety (0.50) > Revenue (0.25) > Mileage (0.15) > Other (0.10)
        """
        # Tier 1: Safety Risk
        r_crit = np.clip(df['critical_jobs'] / 2.0, 0.0, 1.0)
        days_rem = df['days_remaining'].values
        r_cert = np.zeros(len(df))
        cond1 = days_rem <= 0
        cond2 = (days_rem > 0) & (days_rem <= 14)
        cond3 = days_rem > 14
        r_cert[cond1] = np.clip(0.70 + np.abs(days_rem[cond1]) * 0.05, 0.0, 1.0)
        r_cert[cond2] = np.clip(0.30 + ((14 - days_rem[cond2]) / 14.0) * 0.40, 0.0, 1.0)
        r_cert[cond3] = np.clip(np.maximum(0.0, (30 - days_rem[cond3]) / 60.0), 0.0, 1.0)
        r_non_crit = np.clip(df['non_critical_jobs'] / 8.0, 0.0, 1.0)
        R_safety = 0.50 * r_crit + 0.35 * r_cert + 0.15 * r_non_crit

        # Tier 2: Revenue Risk
        r_penalty = np.clip((df['penalty'] - 10000.0) / 38000.0, 0.0, 1.0)
        r_rem_hours = np.clip(df['remaining_hours'] / 450.0, 0.0, 1.0)
        r_revenue = np.clip(df['revenue'] / 32000.0, 0.0, 1.0)
        R_revenue = 0.45 * r_penalty + 0.35 * r_rem_hours + 0.20 * r_revenue

        # Tier 3: Mileage Risk
        r_daily_mileage = np.clip(df['mileage'] / 450.0, 0.0, 1.0)
        r_total_km = np.clip(df['total_km'] / 65000.0, 0.0, 1.0)
        R_mileage = 0.45 * r_daily_mileage + 0.55 * r_total_km

        # Tier 4: Other Factors
        r_days_maint = np.clip(df['days_since_last_maintenance'] / 45.0, 0.0, 1.0)
        r_age = np.clip(df['train_age_years'] / 20.0, 0.0, 1.0)
        r_var = np.clip(np.sqrt(np.maximum(0.0, df['variance'])) / 80.0, 0.0, 1.0)
        r_avg_run = np.clip(df['daily_avg_run'] / 400.0, 0.0, 1.0)
        r_trips = np.clip(df['daily_trips'] / 25.0, 0.0, 1.0)
        R_other = 0.35 * r_days_maint + 0.25 * r_age + 0.15 * r_var + 0.15 * r_avg_run + 0.10 * r_trips

        raw = 0.50 * R_safety + 0.25 * R_revenue + 0.15 * R_mileage + 0.10 * R_other
        return np.round(np.clip(raw, 0.001, 0.999), 4)

    @classmethod
    def compute_temporal_features(cls, history_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes 7-day rolling window averages and 7-day velocity differentials
        for a sequence of operational days sorted by train_id and date.
        """
        df = history_df.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values(['train_id', 'date']).reset_index(drop=True)

        if 'maintenance_priority_score' not in df.columns or df['maintenance_priority_score'].isna().any():
            df['maintenance_priority_score'] = cls.calculate_priority_score(df)

        for col in ['critical_jobs', 'non_critical_jobs', 'daily_avg_run', 'revenue', 'penalty']:
            df[f'rolling_7d_{col}'] = df.groupby('train_id')[col].transform(lambda s: s.rolling(7, min_periods=1).mean())

        df['delta_7d_critical_jobs'] = df.groupby('train_id')['critical_jobs'].transform(lambda s: s.diff(7)).fillna(0)
        df['delta_7d_days_remaining'] = df.groupby('train_id')['days_remaining'].transform(lambda s: s.diff(7)).fillna(0)
        df['delta_7d_total_km'] = df.groupby('train_id')['total_km'].transform(lambda s: s.diff(7)).fillna(0)
        df['delta_7d_priority_score'] = df.groupby('train_id')['maintenance_priority_score'].transform(lambda s: s.diff(7)).fillna(0)
        df['delta_7d_days_since_maint'] = df.groupby('train_id')['days_since_last_maintenance'].transform(lambda s: s.diff(7)).fillna(0)

        return df

    @classmethod
    def extract_features(cls, input_data: Union[pd.DataFrame, List[Dict[str, Any]]]) -> pd.DataFrame:
        """
        Validates and extracts exactly the 24 canonical feature columns in required order.
        """
        if isinstance(input_data, list):
            df = pd.DataFrame(input_data)
        else:
            df = input_data.copy()

        # Check if rolling/delta features are missing but history exists
        missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing_features:
            if 'train_id' in df.columns and len(df) > 1:
                df = cls.compute_temporal_features(df)
            else:
                raise ValueError(f"Missing required model features: {missing_features}")

        return df[FEATURE_COLUMNS]

