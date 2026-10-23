"""
Comprehensive evaluation metrics for regression accuracy and ranking fidelity.
"""
from typing import Dict, Any
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, ndcg_score

def evaluate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df_subset: pd.DataFrame,
    target_col: str
) -> Dict[str, Any]:
    """
    Computes absolute point prediction metrics and daily ranking correlation metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    spearmans = []
    kendalls = []
    rank_errors = []
    top1_list = []
    top3_list = []
    top5_list = []
    ndcg3_list = []
    ndcg5_list = []
    
    temp_df = df_subset.copy()
    temp_df['pred_temp'] = y_pred
    
    for dt, grp in temp_df.groupby('date'):
        act_ranks = grp[target_col].rank(ascending=False, method='min').values
        pred_ranks = grp['pred_temp'].rank(ascending=False, method='min').values
        
        if len(np.unique(act_ranks)) <= 1 or len(np.unique(pred_ranks)) <= 1:
            spearmans.append(0.0)
            kendalls.append(0.0)
        else:
            sp = spearmanr(act_ranks, pred_ranks).statistic
            kt = kendalltau(act_ranks, pred_ranks).statistic
            spearmans.append(0.0 if np.isnan(sp) else float(sp))
            kendalls.append(0.0 if np.isnan(kt) else float(kt))
            
        rank_errors.append(float(np.mean(np.abs(act_ranks - pred_ranks))))
        
        top1_act = set(grp.nlargest(1, target_col)['train_id'])
        top1_pred = set(grp.nlargest(1, 'pred_temp')['train_id'])
        top1_list.append(len(top1_act & top1_pred) / 1.0)
        
        top3_act = set(grp.nlargest(3, target_col)['train_id'])
        top3_pred = set(grp.nlargest(3, 'pred_temp')['train_id'])
        top3_list.append(len(top3_act & top3_pred) / 3.0)
        
        top5_act = set(grp.nlargest(5, target_col)['train_id'])
        top5_pred = set(grp.nlargest(5, 'pred_temp')['train_id'])
        top5_list.append(len(top5_act & top5_pred) / 5.0)
        
        act_scores = np.asarray([grp[target_col].values])
        pred_scores = np.asarray([grp['pred_temp'].values])
        ndcg3_list.append(float(ndcg_score(act_scores, pred_scores, k=3)))
        ndcg5_list.append(float(ndcg_score(act_scores, pred_scores, k=5)))
        
    return {
        'mae': round(float(mae), 5),
        'rmse': round(float(rmse), 5),
        'r2': round(float(r2), 4),
        'daily_spearman': round(float(np.mean(spearmans)), 4),
        'daily_kendall': round(float(np.mean(kendalls)), 4),
        'mean_rank_error': round(float(np.mean(rank_errors)), 2),
        'top1_overlap': round(float(np.mean(top1_list)) * 100, 2),
        'top3_overlap': round(float(np.mean(top3_list)) * 100, 2),
        'top5_overlap': round(float(np.mean(top5_list)) * 100, 2),
        'ndcg3': round(float(np.mean(ndcg3_list)) * 100, 2),
        'ndcg5': round(float(np.mean(ndcg5_list)) * 100, 2)
    }

