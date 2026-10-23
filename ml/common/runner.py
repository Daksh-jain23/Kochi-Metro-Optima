"""
Standardized training execution, CLI argument parsing, and result serialization.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional, Type
import sys
import json
import joblib
import pandas as pd
from .data import load_data, get_chronological_splits, FEATURE_COLUMNS
from .metrics import evaluate_metrics

def parse_horizon_arg(default_horizons: List[int] = [3, 7]) -> List[int]:
    """Parses CLI argument if passed (e.g. `python train.py 3`), else defaults to [3, 7]."""
    if len(sys.argv) > 1:
        return [int(sys.argv[1])]
    return default_horizons

def save_model_results(
    output_dir: Path,
    model_name: str,
    horizon: int,
    metrics: Dict[str, Any],
    model: Optional[Any] = None,
    primary_filename: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    extra_data: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Saves joblib model bundle and result_t{horizon}.json to target output directory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model artifact and any legacy aliases
    if model is not None and primary_filename:
        model_file = output_path / primary_filename
        joblib.dump(model, model_file)
        if aliases:
            for alias in aliases:
                joblib.dump(model, output_path / alias)
        print(f"Saved Model: {model_file}")

    # Prepare JSON structure
    res_data: Dict[str, Any] = {
        "model": model_name,
        "horizon": f"t+{horizon}",
        "metrics": metrics
    }
    if hyperparameters:
        res_data["hyperparameters"] = hyperparameters
    if extra_data:
        res_data.update(extra_data)

    json_file = output_path / f"result_t{horizon}.json"
    with open(json_file, 'w') as f:
        json.dump(res_data, f, indent=2)

    print(f"Saved Results: {json_file}")
    print(json.dumps(metrics, indent=2))

def train_standard_regressor(
    model_factory,
    model_name: str,
    output_dir: Path,
    file_prefix: str,
    params: Optional[Dict[str, Any]] = None,
    horizon: int = 3,
    aliases: Optional[List[str]] = None
) -> None:
    """
    Runs standardized training loop on Train + Val splits and evaluates on Test split.
    """
    print(f"\n==================== TRAINING {model_name.upper()} FOR t+{horizon} ====================")
    df, target_col = load_data(horizon=horizon)
    train_df, val_df, test_df = get_chronological_splits(df)
    
    # Fit model on Train + Val
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    model = model_factory(**params) if params else model_factory()
    model.fit(train_val_df[FEATURE_COLUMNS], train_val_df[target_col])
    
    # Evaluate on untouched Test set
    test_pred = model.predict(test_df[FEATURE_COLUMNS])
    metrics = evaluate_metrics(test_df[target_col].values, test_pred, test_df, target_col)
    
    # Save
    primary_filename = f"{file_prefix}_t{horizon}.joblib"
    alias_filenames = [f"{a}_t{horizon}.joblib" for a in (aliases or [])]
    
    save_model_results(
        output_dir=output_dir,
        model_name=model_name,
        horizon=horizon,
        metrics=metrics,
        model=model,
        primary_filename=primary_filename,
        aliases=alias_filenames,
        hyperparameters=params
    )

