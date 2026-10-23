"""
Shared dataset loader, feature definitions, and chronological splitting.
"""
from pathlib import Path
from typing import List, Tuple
import pandas as pd

# Root project directory resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

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

def load_data(
    horizon: int = 3,
    data_path: str = "ml/data/data_2025/railway_maintenance_365_days.csv"
) -> Tuple[pd.DataFrame, str]:
    """
    Loads raw CSV, constructs the forecast target shifted by -horizon days,
    and engineers 7-day rolling statistics and velocity features.
    """
    resolved_path = PROJECT_ROOT / data_path if not Path(data_path).is_absolute() else Path(data_path)
    df = pd.read_csv(resolved_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['train_id', 'date']).reset_index(drop=True)
    
    target_col = f'target_priority_{horizon}d'
    df[target_col] = df.groupby('train_id')['maintenance_priority_score'].shift(-horizon)
    
    for col in ['critical_jobs', 'non_critical_jobs', 'daily_avg_run', 'revenue', 'penalty']:
        df[f'rolling_7d_{col}'] = df.groupby('train_id')[col].transform(lambda s: s.rolling(7, min_periods=1).mean())
        
    df['delta_7d_critical_jobs'] = df.groupby('train_id')['critical_jobs'].transform(lambda s: s.diff(7)).fillna(0)
    df['delta_7d_days_remaining'] = df.groupby('train_id')['days_remaining'].transform(lambda s: s.diff(7)).fillna(0)
    df['delta_7d_total_km'] = df.groupby('train_id')['total_km'].transform(lambda s: s.diff(7)).fillna(0)
    df['delta_7d_priority_score'] = df.groupby('train_id')['maintenance_priority_score'].transform(lambda s: s.diff(7)).fillna(0)
    df['delta_7d_days_since_maint'] = df.groupby('train_id')['days_since_last_maintenance'].transform(lambda s: s.diff(7)).fillna(0)
    
    valid_df = df.dropna(subset=[target_col]).copy()
    valid_df = valid_df.sort_values(['date', 'train_id']).reset_index(drop=True)
    valid_df['relevance_grade'] = valid_df.groupby('date')[target_col].rank(ascending=True, method='min').astype(int) - 1
    
    return valid_df, target_col

def get_chronological_splits(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset chronologically into Train (Jan-Aug/Sep), Validation (Sep-Oct),
    and Untouched Test (Nov-Dec).
    """
    min_year = df['date'].dt.year.min()
    train_end = pd.to_datetime(f'{min_year}-09-07')
    val_end = pd.to_datetime(f'{min_year}-10-31')
    
    train_df = df[df['date'] <= train_end].copy().sort_values(['date', 'train_id']).reset_index(drop=True)
    val_df = df[(df['date'] > train_end) & (df['date'] <= val_end)].copy().sort_values(['date', 'train_id']).reset_index(drop=True)
    test_df = df[df['date'] > val_end].copy().sort_values(['date', 'train_id']).reset_index(drop=True)
    
    return train_df, val_df, test_df

