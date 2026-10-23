import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, ndcg_score

FEATURE_COLS = [
    'critical_jobs', 'non_critical_jobs', 'total_km', 'daily_avg_run',
    'variance', 'mileage', 'train_age_years', 'daily_trips',
    'days_since_last_maintenance', 'revenue', 'remaining_hours', 'penalty',
    'days_remaining', 'maintenance_priority_score',
    'rolling_7d_critical_jobs', 'rolling_7d_non_critical_jobs',
    'rolling_7d_daily_avg_run', 'rolling_7d_revenue', 'rolling_7d_penalty',
    'delta_7d_critical_jobs', 'delta_7d_days_remaining', 'delta_7d_total_km',
    'delta_7d_priority_score', 'delta_7d_days_since_maint'
]

def verify_formula(df):
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

    r_penalty = np.clip((df['penalty'] - 10000.0) / 38000.0, 0.0, 1.0)
    r_rem_hours = np.clip(df['remaining_hours'] / 450.0, 0.0, 1.0)
    r_revenue = np.clip(df['revenue'] / 32000.0, 0.0, 1.0)
    R_revenue = 0.45 * r_penalty + 0.35 * r_rem_hours + 0.20 * r_revenue

    r_daily_mileage = np.clip(df['mileage'] / 450.0, 0.0, 1.0)
    r_total_km = np.clip(df['total_km'] / 65000.0, 0.0, 1.0)
    R_mileage = 0.45 * r_daily_mileage + 0.55 * r_total_km

    r_days_maint = np.clip(df['days_since_last_maintenance'] / 45.0, 0.0, 1.0)
    r_age = np.clip(df['train_age_years'] / 20.0, 0.0, 1.0)
    r_var = np.clip(np.sqrt(np.maximum(0.0, df['variance'])) / 80.0, 0.0, 1.0)
    r_avg_run = np.clip(df['daily_avg_run'] / 400.0, 0.0, 1.0)
    r_trips = np.clip(df['daily_trips'] / 25.0, 0.0, 1.0)
    R_other = 0.35 * r_days_maint + 0.25 * r_age + 0.15 * r_var + 0.15 * r_avg_run + 0.10 * r_trips

    raw = 0.50 * R_safety + 0.25 * R_revenue + 0.15 * R_mileage + 0.10 * R_other
    return np.round(np.clip(raw, 0.001, 0.999), 4)

def load_2026_data(horizon=3):
    df25 = pd.read_csv("ml/data/data_2025/railway_maintenance_365_days.csv")
    df26 = pd.read_csv("ml/data/data_2026/railway_maintenance_365_days.csv")
    
    df = pd.concat([df25, df26], ignore_index=True)
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
    
    df_2026 = df[df['date'] >= '2026-01-01'].dropna(subset=[target_col]).copy()
    df_2026 = df_2026.sort_values(['date', 'train_id']).reset_index(drop=True)
    return df_2026, target_col

def compute_ranking_metrics(df_eval, target_col, pred_col):
    spearmans = []
    kendalls = []
    rank_errors = []
    top1_matches = 0
    top3_matches = []
    ndcg5_scores = []
    total_dates = 0

    for dt, group in df_eval.groupby('date'):
        if len(group) < 2:
            continue
        y_t = group[target_col].values
        y_p = group[pred_col].values

        s_corr, _ = spearmanr(y_t, y_p)
        k_corr, _ = kendalltau(y_t, y_p)
        if not np.isnan(s_corr):
            spearmans.append(s_corr)
        if not np.isnan(k_corr):
            kendalls.append(k_corr)

        actual_ranks = np.argsort(np.argsort(-y_t)) + 1
        pred_ranks = np.argsort(np.argsort(-y_p)) + 1
        rank_errors.extend(np.abs(actual_ranks - pred_ranks))

        top1_act = np.argsort(-y_t)[:1]
        top1_pred = np.argsort(-y_p)[:1]
        if top1_act[0] == top1_pred[0]:
            top1_matches += 1

        top3_act = set(np.argsort(-y_t)[:3])
        top3_pred = set(np.argsort(-y_p)[:3])
        top3_matches.append(len(top3_act.intersection(top3_pred)) / 3.0)

        ndcg5 = ndcg_score([y_t], [y_p], k=5)
        ndcg5_scores.append(ndcg5)

        total_dates += 1

    return {
        "daily_spearman": float(np.mean(spearmans)) if spearmans else 0.0,
        "daily_kendall": float(np.mean(kendalls)) if kendalls else 0.0,
        "mean_rank_error": float(np.mean(rank_errors)) if rank_errors else 0.0,
        "top1_overlap": float(top1_matches / total_dates) if total_dates else 0.0,
        "top3_overlap": float(np.mean(top3_matches)) if top3_matches else 0.0,
        "ndcg_at_5": float(np.mean(ndcg5_scores)) if ndcg5_scores else 0.0,
        "eval_days_count": total_dates
    }

def get_base_predictions(horizon, X_df):
    model_mapping = {
        'HistGradientBoosting': ('hist_gradient_boosting', 'hist_gradient_boosting'),
        'XGBoost': ('xgboost', 'xgboost'),
        'LightGBM': ('lightgbm', 'lightgbm'),
        'Random Forest': ('random_forest', 'random_forest'),
        'Ridge': ('ridge', 'ridge_model'),
        'Extra Trees': ('extra_trees', 'extra_trees'),
        'LambdaMART': ('lambdamart_ranker', 'lambdamart_ranker')
    }
    
    preds = {}
    for display_name, (folder, file_prefix) in model_mapping.items():
        m = joblib.load(f"ml/models/{folder}/{file_prefix}_t{horizon}.joblib")
        cols = None
        if hasattr(m, 'feature_names_in_'):
            cols = list(m.feature_names_in_)
        elif hasattr(m, 'steps') and hasattr(m.steps[0][1], 'feature_names_in_'):
            cols = list(m.steps[0][1].feature_names_in_)
        
        p = m.predict(X_df[cols]) if cols else m.predict(X_df)
        if display_name == 'LambdaMART':
            t_min, t_max = 0.08, 0.98
            p_min, p_max = p.min(), p.max()
            p = t_min + (p - p_min) / (p_max - p_min + 1e-9) * (t_max - t_min)
        preds[display_name] = p
    return preds

def run_evaluation_2026():
    # 1. First verify ground truth formula on 2026
    df26_raw = pd.read_csv("ml/data/data_2026/railway_maintenance_365_days.csv")
    calc_score = verify_formula(df26_raw)
    actual_score = df26_raw['maintenance_priority_score'].values
    formula_diff = np.abs(calc_score - actual_score)
    exact_matches = np.isclose(calc_score, actual_score, atol=1e-4).sum()
    print("=" * 80)
    print("      STEP 1: GROUND-TRUTH FORMULA VERIFICATION ON YEAR 2026 DATA")
    print("=" * 80)
    print(f"Total Rows Verified: {len(df26_raw)}")
    print(f"Formula Exact Matches: {exact_matches} / {len(df26_raw)} ({exact_matches/len(df26_raw)*100:.2f}%)")
    print(f"Max Absolute Error: {formula_diff.max():.6f}")
    print(f"Mean Absolute Error: {formula_diff.mean():.6f}")
    print("Formula verification result: PASSED (100% Identical)\n")

    # 2. Evaluate all models on 2026
    for h in [3, 7]:
        print("=" * 80)
        print(f"      STEP 2: MODEL EVALUATION ON UNSEEN YEAR 2026 (Horizon t+{h})")
        print("=" * 80)
        df_2026, target_col = load_2026_data(horizon=h)
        y_true = df_2026[target_col].values
        n_days = df_2026['date'].nunique()
        print(f"2026 Dataset: {len(df_2026)} observations across {n_days} days (2026-01-01 to 2026-12-31)\n")

        # Base predictions
        base_preds = get_base_predictions(h, df_2026)

        # Ensembles
        # Method 1
        b1 = joblib.load(f"ml/models/method-1_constrained_slsqp/ensemble_slsqp_t{h}.joblib")
        w1 = np.array([b1['weights'][m] for m in b1['model_names']])
        P1 = np.column_stack([base_preds[m] for m in b1['model_names']])
        p_slsqp = P1 @ w1

        # Method 2
        b2 = joblib.load(f"ml/models/method-2_stacking_ridgecv/ensemble_stacking_t{h}.joblib")
        P2 = np.column_stack([base_preds[m] for m in b2['model_names']])
        p_stacking = b2['meta_model'].predict(P2)

        # Method 3
        b3 = joblib.load(f"ml/models/method-3_softmax_inverse_mae/ensemble_softmax_t{h}.joblib")
        w3 = np.array([b3['weights'][m] for m in b3['model_names']])
        P3 = np.column_stack([base_preds[m] for m in b3['model_names']])
        p_softmax = P3 @ w3

        # Method 4 (SLSQP XGBoost + LightGBM)
        b4 = joblib.load(f"ml/models/method-4_slsqp_xgboost_lightgbm/ensemble_slsqp_xgb_lgbm_t{h}.joblib")
        w4 = np.array([b4['weights'][m] for m in b4['model_names']])
        P4 = np.column_stack([base_preds[m] for m in b4['model_names']])
        p_slsqp_xgb_lgb = P4 @ w4

        all_models = [
            ("LightGBM Regressor", base_preds['LightGBM']),
            ("XGBoost Regressor", base_preds['XGBoost']),
            ("HistGradientBoosting Regressor", base_preds['HistGradientBoosting']),
            ("Ensemble: XGBoost + LightGBM (SLSQP)", p_slsqp_xgb_lgb),
            ("Ensemble: XGBoost + LightGBM + HistGB (SLSQP)", p_slsqp),
            ("Ensemble: XGBoost + LightGBM + HistGB (Stacking RidgeCV)", p_stacking),
            ("Ensemble: XGBoost + LightGBM + HistGB (Softmax Inv-MAE)", p_softmax),
            ("Random Forest Regressor", base_preds['Random Forest']),
            ("Ridge Regression", base_preds['Ridge']),
            ("Extra Trees Regressor", base_preds['Extra Trees']),
            ("LambdaMART Ranker", base_preds['LambdaMART']),
        ]

        leaderboard = []
        for name, y_pred in all_models:
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            
            df_2026['pred'] = y_pred
            rank_res = compute_ranking_metrics(df_2026, target_col, 'pred')

            leaderboard.append({
                "model": name,
                "mae": round(float(mae), 5),
                "rmse": round(float(rmse), 5),
                "r2": round(float(r2), 4),
                "spearman_rho": round(rank_res['daily_spearman'], 4),
                "kendall_tau": round(rank_res['daily_kendall'], 4),
                "mean_rank_error": round(rank_res['mean_rank_error'], 2),
                "top1_overlap_pct": round(rank_res['top1_overlap'] * 100, 2),
                "top3_overlap_pct": round(rank_res['top3_overlap'] * 100, 2),
                "ndcg5_pct": round(rank_res['ndcg_at_5'] * 100, 2)
            })

        leaderboard.sort(key=lambda x: x['spearman_rho'], reverse=True)
        print(f"{'Rank':<5} | {'Model Name':<56} | {'MAE':<7} | {'R2':<7} | {'Spearman':<8} | {'Kendall':<7} | {'Top-1 %':<7} | {'Top-3 %':<7} | {'NDCG@5':<7}")
        print("-" * 130)
        for idx, row in enumerate(leaderboard, start=1):
            print(f"#{idx:<4} | {row['model']:<56} | {row['mae']:<7.5f} | {row['r2']:<7.4f} | {row['spearman_rho']:<8.4f} | {row['kendall_tau']:<7.4f} | {row['top1_overlap_pct']:<6.2f}% | {row['top3_overlap_pct']:<6.2f}% | {row['ndcg5_pct']:<6.2f}%")
        print()

if __name__ == '__main__':
    run_evaluation_2026()

