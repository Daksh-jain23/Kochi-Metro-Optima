"""
Cached Model Loader for Backend Inference.
Ensures zero redundant disk I/O by caching loaded joblib artifacts in memory.
"""
import logging
from typing import Dict, Any, Optional
import joblib
from ..config import MODEL_PATHS

logger = logging.getLogger(__name__)

class ModelLoader:
    _instance: Optional["ModelLoader"] = None
    _cache: Dict[str, Any] = {}

    def __new__(cls) -> "ModelLoader":
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._cache = {}
        return cls._instance

    @classmethod
    def get_model(cls, horizon: str = "t3", model_type: str = "ensemble") -> Any:
        """
        Retrieves a loaded model bundle from cache, or loads it from disk.
        
        Args:
            horizon: 't3' or 't7'
            model_type: 'ensemble', 'xgboost', or 'lightgbm'
        """
        horizon_key = horizon.lower().replace("+", "")
        if horizon_key not in MODEL_PATHS:
            raise ValueError(f"Unsupported horizon '{horizon}'. Must be 't3' or 't7'.")

        type_key = model_type.lower()
        if type_key not in MODEL_PATHS[horizon_key]:
            raise ValueError(
                f"Unsupported model_type '{model_type}' for horizon '{horizon}'. "
                f"Available: {list(MODEL_PATHS[horizon_key].keys())}"
            )

        cache_key = f"{horizon_key}_{type_key}"
        if cache_key not in cls._cache:
            model_path = MODEL_PATHS[horizon_key][type_key]
            if not model_path.exists():
                raise FileNotFoundError(f"Model artifact not found at: {model_path}")
            
            logger.info(f"Loading model '{cache_key}' from {model_path}...")
            cls._cache[cache_key] = joblib.load(model_path)
            
        return cls._cache[cache_key]

    @classmethod
    def clear_cache(cls) -> None:
        """Clears memory cache."""
        cls._cache.clear()

