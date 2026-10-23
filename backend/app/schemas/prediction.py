"""
Data structures for model predictions, fleet rankings, and depot dispatch recommendations.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class TrainPriorityPrediction:
    """Predicted future maintenance priority and depot recommendation for one train."""
    train_id: str
    forecast_score: float
    current_score: float
    rank: int
    status: str
    horizon: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_id": self.train_id,
            "forecast_score": self.forecast_score,
            "current_score": self.current_score,
            "rank": self.rank,
            "status": self.status,
            "horizon": self.horizon,
            "details": self.details,
        }

@dataclass
class FleetPriorityResponse:
    """Full fleet ranked maintenance forecast ready for scheduling and optimization."""
    horizon: str
    model_used: str
    evaluation_date: Optional[str]
    total_trains: int
    critical_depot_count: int
    inspection_bay_count: int
    active_service_count: int
    predictions: List[TrainPriorityPrediction] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizon": self.horizon,
            "model_used": self.model_used,
            "evaluation_date": self.evaluation_date,
            "total_trains": self.total_trains,
            "critical_depot_count": self.critical_depot_count,
            "inspection_bay_count": self.inspection_bay_count,
            "active_service_count": self.active_service_count,
            "predictions": [p.to_dict() for p in self.predictions],
        }

