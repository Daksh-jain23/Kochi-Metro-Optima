"""
Data structures for incoming operational train telemetry and fleet state.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class TrainOperationalState:
    """Individual metro train operational snapshot on day t."""
    train_id: str
    date: str
    days_remaining: float
    critical_jobs: float
    non_critical_jobs: float
    revenue: float
    remaining_hours: float
    penalty: float
    total_km: float
    daily_avg_run: float
    variance: float
    mileage: float
    train_age_years: float
    daily_trips: float
    days_since_last_maintenance: float
    maintenance_priority_score: Optional[float] = None
    
    # Optional historical features if pre-computed by client
    rolling_7d_critical_jobs: Optional[float] = None
    rolling_7d_non_critical_jobs: Optional[float] = None
    rolling_7d_daily_avg_run: Optional[float] = None
    rolling_7d_revenue: Optional[float] = None
    rolling_7d_penalty: Optional[float] = None
    delta_7d_critical_jobs: Optional[float] = None
    delta_7d_days_remaining: Optional[float] = None
    delta_7d_total_km: Optional[float] = None
    delta_7d_priority_score: Optional[float] = None
    delta_7d_days_since_maint: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}

@dataclass
class FleetOperationalSnapshot:
    """Snapshot of the full fleet (e.g. 25 trains) on a specific operational day."""
    date: str
    trains: List[TrainOperationalState] = field(default_factory=list)

