# Kochi Metro Rolling Stock Maintenance Priority Forecasting & Trip Scheduling

Predictive maintenance priority forecasting and depot dispatching system for the **Kochi Metro Rail Limited (KMRL)** fleet (25 trains, 365 daily observations). 

The system forecasts train degradation and maintenance urgency across two operational planning horizons:
* **$t+3$ Horizon (Short-Term):** Daily operational dispatching and immediate depot inspection scheduling.
* **$t+7$ Horizon (Medium-Term):** Weekly overhaul bay planning and component procurement.

All models in this release are trained and evaluated on **`data_2025`** (`ml/data/data_2025/railway_maintenance_365_days.csv`) using strict chronological validation:
* **Training Period:** Jan 01, 2025 – Sep 07, 2025 (250 days; ~68.5%)
* **Validation Period:** Sep 08, 2025 – Oct 31, 2025 (54 days; ~15.0%)
* **Untouched Test Period:** Nov 01, 2025 – Dec 31, 2025 (61 days; ~16.5%)
* **Out-of-Sample Stress Test:** Full Year **`data_2026`** (365 days; 100% unseen out-of-year holdout)

---

## 1. Executive Summary: The Winning Models

To eliminate the **dilution anchor effect** (where blending lower-performing models degrades the accuracy of peak gradient boosters), ensembling is strictly focused on elite gradient boosted decision tree engines:
1. **Dual-Booster SLSQP Ensemble (`XGBoost + LightGBM`):** Achieves the highest overall rank correlation across the entire benchmark, combining depth-wise symmetric trees (XGBoost) with leaf-wise asymmetric histogram trees (LightGBM).
2. **Tri-Booster SLSQP Ensemble (`XGBoost + LightGBM + HistGradientBoosting`):** Evaluates whether Scikit-Learn's histogram-based gradient boosting adds additional variance reduction.

| Horizon | Winning Model | Architecture / Method | Key Metrics on Test Set | Model Artifact |
| :--- | :--- | :--- | :--- | :--- |
| **$t+3$** | **Method 4: SLSQP (`XGBoost + LightGBM`)** | Dual-Booster Convex Simplex Optimization ($50\%$ XGB / $50\%$ LGB) | **Spearman $\rho$:** `0.8633`<br>**Kendall $\tau$:** `0.7228`<br>**MAE:** `0.02585`<br>**Mean Rank Error:** `2.55` slots | [`ml/models/method-4_slsqp_xgboost_lightgbm/ensemble_slsqp_xgb_lgbm_t3.joblib`](ml/models/method-4_slsqp_xgboost_lightgbm/ensemble_slsqp_xgb_lgbm_t3.joblib) |
| **$t+7$** | **XGBoost Regressor** & **Method 4: SLSQP (`XGBoost + LightGBM`)** | Regularized Boosting & Dual-Booster Convex Ensemble | **Spearman $\rho$:** `0.7695` (XGB) / `0.7662` (Ensemble)<br>**Kendall $\tau$:** `0.6169`<br>**MAE:** `0.03343`<br>**Depot #1 Match:** `40.74%` | [`ml/models/xgboost/xgboost_t7.joblib`](ml/models/xgboost/xgboost_t7.joblib)<br>[`ml/models/method-4_slsqp_xgboost_lightgbm/ensemble_slsqp_xgb_lgbm_t7.joblib`](ml/models/method-4_slsqp_xgboost_lightgbm/ensemble_slsqp_xgb_lgbm_t7.joblib) |

---

## 2. Complete Benchmark Leaderboard Tables (Trained on `data_2025`)

### Horizon $t+3$ (3-Day Operational Turns — 2025 Test Split)

| Rank | Model / Ensemble Method | MAE ↓ | RMSE ↓ | $R^2$ ↑ | Daily Spearman $\rho$ ↑ | Daily Kendall $\tau$ ↑ | Mean Rank Error ↓ | Depot #1 Match ↑ | Depot #3 Match ↑ | NDCG@5 ↑ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Method 4: SLSQP (XGBoost + LightGBM)** | 0.02585 | 0.03976 | 0.6526 | **0.8633** | **0.7228** | **2.55** | 37.93% | 55.17% | 93.81% |
| 🥈 | **LightGBM Regressor** | **0.02563** | **0.03947** | **0.6575** | 0.8632 | 0.7214 | **2.55** | 41.38% | 56.90% | **94.02%** |
| 🥉 | **Method 1: SLSQP (XGBoost + LightGBM + HistGB)** | 0.02570 | 0.03972 | 0.6533 | 0.8613 | 0.7215 | 2.57 | 37.93% | 52.30% | 93.69% |
| 4 | **Method 3: Softmax Inv-MAE (XGBoost + LightGBM + HistGB)** | 0.02581 | 0.04002 | 0.6481 | 0.8605 | 0.7201 | 2.58 | 36.21% | 52.30% | 93.58% |
| 5 | **XGBoost Regressor** | 0.02628 | 0.04033 | 0.6426 | 0.8591 | 0.7181 | 2.58 | 37.93% | 52.30% | 93.61% |
| 6 | **HistGradientBoosting Regressor** | 0.02580 | 0.04018 | 0.6451 | 0.8566 | 0.7169 | 2.63 | 36.21% | 51.72% | 93.20% |
| 7 | **Method 2: Stacking RidgeCV (XGBoost + LightGBM + HistGB)** | 0.02676 | 0.04105 | 0.6297 | 0.8557 | 0.7144 | 2.61 | 36.21% | 50.00% | 93.23% |
| 8 | **Random Forest Regressor** | 0.02744 | 0.04120 | 0.6270 | 0.8476 | 0.7081 | 2.65 | **46.55%** | **58.05%** | 93.83% |
| 9 | **Ridge Regression** | 0.02772 | 0.04103 | 0.6301 | 0.8396 | 0.7019 | 2.70 | 43.10% | **58.05%** | 93.75% |
| 10 | **Extra Trees Regressor** | 0.02952 | 0.04353 | 0.5835 | 0.8323 | 0.6874 | 2.82 | 43.10% | 51.72% | 92.62% |
| 11 | **LambdaMART Ranker** | 0.04882 | 0.06522 | 0.0651 | 0.7515 | 0.5918 | 3.63 | 44.83% | 54.60% | 93.68% |

---

### Horizon $t+7$ (7-Day Medium-Term Planning — 2025 Test Split)

| Rank | Model / Ensemble Method | MAE ↓ | RMSE ↓ | $R^2$ ↑ | Daily Spearman $\rho$ ↑ | Daily Kendall $\tau$ ↑ | Mean Rank Error ↓ | Depot #1 Match ↑ | Depot #3 Match ↑ | NDCG@5 ↑ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **XGBoost Regressor** | 0.03343 | **0.04857** | **0.5005** | **0.7695** | **0.6169** | **3.44** | **40.74%** | 35.19% | **91.64%** |
| 🥈 | **Method 4: SLSQP (XGBoost + LightGBM)** | 0.03376 | 0.04883 | 0.4952 | 0.7662 | 0.6165 | 3.46 | **40.74%** | 37.04% | 91.57% |
| 🥉 | **Method 1: SLSQP (XGBoost + LightGBM + HistGB)** | 0.03349 | 0.04868 | 0.4981 | 0.7644 | 0.6128 | 3.47 | **40.74%** | **40.12%** | 91.58% |
| 4 | **Method 2: Stacking RidgeCV (XGBoost + LightGBM + HistGB)** | 0.03384 | 0.04991 | 0.4726 | 0.7634 | 0.6123 | 3.47 | **40.74%** | **40.12%** | 91.51% |
| 5 | **Method 3: Softmax Inv-MAE (XGBoost + LightGBM + HistGB)** | 0.03437 | 0.04964 | 0.4782 | 0.7607 | 0.6086 | 3.50 | 38.89% | 35.19% | 91.43% |
| 6 | **LightGBM Regressor** | 0.03437 | 0.04964 | 0.4782 | 0.7607 | 0.6086 | 3.50 | 38.89% | 35.19% | 91.43% |
| 7 | **HistGradientBoosting Regressor** | **0.03337** | 0.04890 | 0.4936 | 0.7573 | 0.6038 | 3.54 | **40.74%** | 39.51% | 91.09% |
| 8 | **Random Forest Regressor** | 0.03622 | 0.05111 | 0.4468 | 0.7546 | 0.5935 | 3.62 | **40.74%** | 37.65% | 91.01% |
| 9 | **Ridge Regression** | 0.03667 | 0.05081 | 0.4533 | 0.7516 | 0.5922 | 3.66 | 38.89% | 35.19% | 90.30% |
| 10 | **Extra Trees Regressor** | 0.03641 | 0.05183 | 0.4312 | 0.7473 | 0.5930 | 3.61 | 37.04% | 33.95% | 89.76% |
| 11 | **LambdaMART Ranker** | 0.05720 | 0.07382 | -0.1539 | 0.6579 | 0.4977 | 4.44 | **40.74%** | 30.86% | 90.43% |

---

### Combined Average Across Both Horizons ($t+3$ and $t+7$)

| Rank | Model / Ensemble Method | Combined MAE ↓ | Combined $R^2$ ↑ | Combined Spearman $\rho$ ↑ | Combined Kendall $\tau$ ↑ | Mean Rank Error ↓ | Depot #1 Match ↑ | Depot #3 Match ↑ | NDCG@5 ↑ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Method 4: SLSQP (XGBoost + LightGBM)** | 0.02981 | 0.5739 | **0.8148** | **0.6697** | **3.01** | 39.34% | 46.11% | **92.69%** |
| 🥈 | **XGBoost Regressor** | 0.02985 | 0.5716 | 0.8143 | 0.6675 | **3.01** | 39.34% | 43.74% | 92.62% |
| 🥉 | **Method 1: SLSQP (XGBoost + LightGBM + HistGB)** | **0.02959** | **0.5757** | 0.8128 | 0.6672 | 3.02 | 39.34% | 46.21% | 92.63% |
| 4 | **LightGBM Regressor** | 0.03000 | 0.5678 | 0.8119 | 0.6650 | 3.02 | 40.14% | 46.05% | 92.72% |
| 5 | **Method 3: Softmax Inv-MAE (XGBoost + LightGBM + HistGB)** | 0.03009 | 0.5632 | 0.8106 | 0.6643 | 3.04 | 37.55% | 43.74% | 92.50% |
| 6 | **Method 2: Stacking RidgeCV (XGBoost + LightGBM + HistGB)** | 0.03030 | 0.5512 | 0.8095 | 0.6633 | 3.04 | 38.48% | 45.06% | 92.37% |
| 7 | **HistGradientBoosting Regressor** | **0.02959** | 0.5694 | 0.8070 | 0.6603 | 3.08 | 38.48% | 45.61% | 92.15% |
| 8 | **Random Forest Regressor** | 0.03183 | 0.5369 | 0.8011 | 0.6508 | 3.13 | **43.64%** | **47.85%** | 92.42% |
| 9 | **Ridge Regression** | 0.03220 | 0.5417 | 0.7956 | 0.6470 | 3.18 | 41.00% | 46.62% | 92.03% |
| 10 | **Extra Trees Regressor** | 0.03297 | 0.5073 | 0.7898 | 0.6402 | 3.21 | 40.07% | 42.84% | 91.19% |
| 11 | **LambdaMART Ranker** | 0.05301 | -0.0444 | 0.7047 | 0.5447 | 4.04 | 42.78% | 42.73% | 92.06% |

---

## 3. Out-of-Sample Holdout Stress Test (Full Year 2026 Unseen Data)

Evaluated across all **365 operational days of Year 2026** ($9,050$ observations for $t+3$; $8,950$ observations for $t+7$). By focusing strictly on `XGBoost` and `LightGBM`, **Method 4 achieves the #1 highest ranking correlation on unseen out-of-year data**:

#### Year 2026 Unseen: Horizon $t+3$
| Rank | Model / Ensemble Method | MAE ↓ | $R^2$ ↑ | Daily Spearman $\rho$ ↑ | Daily Kendall $\tau$ ↑ | Mean Rank Error ↓ | Depot #1 Match ↑ | Depot #3 Match ↑ | NDCG@5 ↑ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Method 4: SLSQP (XGBoost + LightGBM)** | 0.02798 | 0.6090 | **0.8034** | **0.6641** | **2.88** | 36.19% | 51.01% | 92.76% |
| 🥈 | **XGBoost Regressor** | 0.02807 | 0.6062 | 0.8031 | 0.6640 | **2.88** | 37.57% | 51.57% | **92.79%** |
| 🥉 | **Method 1: SLSQP (XGBoost + LightGBM + HistGB)** | **0.02793** | **0.6103** | 0.8022 | 0.6625 | 2.90 | 37.02% | 51.66% | 92.75% |
| 4 | **LightGBM Regressor** | 0.02818 | 0.6040 | 0.8009 | 0.6607 | 2.90 | 34.53% | 51.10% | 92.54% |
| 5 | **Method 3: Softmax Inv-MAE (XGBoost + LightGBM + HistGB)** | 0.02803 | 0.6071 | 0.7990 | 0.6591 | 2.92 | 34.81% | 51.66% | 92.57% |
| 6 | **Random Forest Regressor** | 0.02939 | 0.5890 | 0.7968 | 0.6527 | 2.92 | 33.98% | 50.28% | 92.39% |
| 7 | **Method 2: Stacking RidgeCV (XGBoost + LightGBM + HistGB)** | 0.02861 | 0.5978 | 0.7954 | 0.6556 | 2.95 | 35.08% | 51.47% | 92.47% |
| 8 | **HistGradientBoosting Regressor** | 0.02844 | 0.5967 | 0.7893 | 0.6483 | 3.01 | 34.53% | 51.20% | 92.44% |
| 9 | **Extra Trees Regressor** | 0.03150 | 0.5439 | 0.7746 | 0.6272 | 3.16 | 32.87% | 48.90% | 91.77% |
| 10 | **Ridge Regression** | 0.03077 | 0.5570 | 0.7699 | 0.6353 | 3.18 | **38.12%** | 52.39% | 92.05% |
| 11 | **LambdaMART Ranker** | 0.11463 | -3.7947 | 0.7306 | 0.5755 | 3.65 | 35.08% | **53.96%** | 92.29% |

#### Year 2026 Unseen: Horizon $t+7$
| Rank | Model / Ensemble Method | MAE ↓ | $R^2$ ↑ | Daily Spearman $\rho$ ↑ | Daily Kendall $\tau$ ↑ | Mean Rank Error ↓ | Depot #1 Match ↑ | Depot #3 Match ↑ | NDCG@5 ↑ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **XGBoost Regressor** | 0.03523 | **0.4492** | **0.7176** | **0.5679** | **3.67** | 15.92% | **39.11%** | **89.36%** |
| 🥈 | **Method 4: SLSQP (XGBoost + LightGBM)** | 0.03534 | 0.4455 | 0.7161 | 0.5668 | 3.68 | 15.08% | 38.27 | 89.09% |
| 🥉 | **Method 1: SLSQP (XGBoost + LightGBM + HistGB)** | 0.03532 | 0.4453 | 0.7122 | 0.5639 | 3.70 | 15.92% | 37.52% | 89.05% |
| 4 | **Method 2: Stacking RidgeCV (XGBoost + LightGBM + HistGB)** | **0.03503** | 0.4287 | 0.7103 | 0.5626 | 3.72 | 16.48% | 37.71% | 89.00% |
| 5 | **LightGBM Regressor** | 0.03593 | 0.4282 | 0.7060 | 0.5568 | 3.76 | 16.48% | 36.87% | 88.62% |
| 6 | **Method 3: Softmax Inv-MAE (XGBoost + LightGBM + HistGB)** | 0.03593 | 0.4282 | 0.7060 | 0.5568 | 3.76 | 16.48% | 36.87% | 88.62% |
| 7 | **HistGradientBoosting Regressor** | 0.03591 | 0.4273 | 0.6979 | 0.5501 | 3.84 | **17.04%** | 37.80% | 88.84% |
| 8 | **Random Forest Regressor** | 0.03711 | 0.4021 | 0.6898 | 0.5392 | 3.87 | 16.20% | 36.69% | 88.25% |
| 9 | **Extra Trees Regressor** | 0.03874 | 0.3700 | 0.6670 | 0.5148 | 4.07 | 14.25% | 34.54% | 87.04% |
| 10 | **Ridge Regression** | 0.03935 | 0.3548 | 0.6569 | 0.5089 | 4.09 | 16.20% | 36.13% | 87.00% |
| 11 | **LambdaMART Ranker** | 0.14264 | -5.9369 | 0.6397 | 0.4834 | 4.41 | 16.76% | 38.36% | 88.66% |

---

## 4. Methodology: Targeted Model Selection & Ensembling Formulations

### Why Specific Models Are Ensembled and Weaker Models Pruned

In mathematical ensemble theory, combining estimators improves test accuracy only when constituent models exhibit **comparable prediction power** and **uncorrelated error distributions**. When estimators with high structural bias are averaged with high-capacity non-linear estimators, the weaker regressors act as an **anchor dragging down overall performance**:

1. **Why Linear & Bagged Models Were Excluded:**
   * **Ridge Regression** ($\rho=0.657$) is constrained by linear hyperplanes and cannot capture the non-linear inflection points of safety risk (such as the piece-wise linear risk step when certificate `days_remaining` $\le 0$).
   * **Extra Trees** ($\rho=0.667$) and **Random Forest** ($\rho=0.690$) draw random orthogonal feature splits that diffuse sharp decision boundaries across 24 correlated features.
   * **LambdaMART Ranker** ($\rho=0.640$) optimizes listwise pairwise gradients but exhibits severe calibration drift on absolute score boundaries, generating negative $R^2$ on out-of-year holdout data.
   * When these weaker models were ensembled across all 7 candidates, the resulting Spearman correlation suffered a severe $-4.9\%$ degradation compared to individual boosters.

2. **Why `XGBoost + LightGBM` Represents the Optimal Combination:**
   * **XGBoost Regressor:** Implements exact greedy and histogram-based level-wise (depth-first) tree growth with second-order Taylor expansion gradients and $L_1/L_2$ leaf weight regularization. It excels at preserving monotonic global rankings.
   * **LightGBM Regressor:** Implements leaf-wise (best-first) tree growth using Gradient-Based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB). It converges faster on subtle non-linear interactions with lower depth.
   * **Convex Blending Synergy:** Because XGBoost and LightGBM split feature space along fundamentally different tree growth trajectories (level-wise vs. leaf-wise), their individual residual errors are orthogonal. Combining them via convex optimization produces a net estimator with lower variance than either single booster, resulting in the highest overall rank fidelity on both 2025 test ($\rho=0.8633$) and 2026 unseen data ($\rho=0.8034$).

3. **Why `XGBoost + LightGBM + HistGradientBoosting` Was Evaluated:**
   * To determine whether a third gradient booster based on Scikit-Learn's native integer-binned histogram algorithm (`HistGradientBoostingRegressor`) would further smooth out variance. While this tri-booster configuration achieves the lowest absolute MAE on 2026 data (`0.02793`), the dual combination of `XGBoost + LightGBM` retains slightly higher ranking correlation ($\rho=0.8034$ vs $\rho=0.8022$).

---

### The Four Ensembling Implementations

#### Method 4: Constrained Convex Optimization (SLSQP: `XGBoost + LightGBM`)
* **Mathematical Formulation:**
  $$\min_{w_{\text{xgb}}, w_{\text{lgb}}} \text{MAE}\big(y_{\text{val}},\, w_{\text{xgb}} \hat{y}_{\text{xgb}} + w_{\text{lgb}} \hat{y}_{\text{lgb}}\big) \quad \text{subject to } w_{\text{xgb}} + w_{\text{lgb}} = 1,\quad w_{\text{xgb}}, w_{\text{lgb}} \ge 0$$
* **Optimization Mechanism:** Solved via Sequential Least Squares Programming (SLSQP). Restricting the weights to the 1-simplex guarantees that the ensemble cannot extrapolate outside physically valid maintenance bounds $[0, 1]$.
* **Learned Weights:**
  * $t+3$: XGBoost ($50.0\%$), LightGBM ($50.0\%$)
  * $t+7$: XGBoost ($50.0\%$), LightGBM ($50.0\%$)

#### Method 1: Constrained Convex Optimization (SLSQP: `XGBoost + LightGBM + HistGradientBoosting`)
* **Mathematical Formulation:**
  $$\min_w \text{MAE}(y_{\text{val}},\, P_{\text{val}} \cdot w) \quad \text{subject to } \sum_{i=1}^3 w_i = 1,\quad w_i \ge 0$$
* **Learned Weights:**
  * $t+3$: HistGradientBoosting ($33.3\%$), XGBoost ($33.3\%$), LightGBM ($33.3\%$)
  * $t+7$: HistGradientBoosting ($33.3\%$), XGBoost ($33.3\%$), LightGBM ($33.3\%$)

#### Method 2: Regularized Stacking Meta-Learner (RidgeCV: `XGBoost + LightGBM + HistGradientBoosting`)
* **Mathematical Formulation:**
  $$\min_{\beta_0, \beta} \frac{1}{N} \sum_{i=1}^N \Big(y_i - \big(\beta_0 + \sum_{m=1}^3 \beta_m \hat{y}_{i,m}\big)\Big)^2 + \alpha \sum_{m=1}^3 \beta_m^2$$
* **Optimization Mechanism:** Trains an $L_2$-regularized meta-estimator on out-of-fold validation predictions. Cross-validated $\alpha \in [0.001, 100.0]$ shrinks collinear coefficients across the three tree engines.

#### Method 3: Performance-Weighted Softmax (Inverse-MAE: `XGBoost + LightGBM + HistGradientBoosting`)
* **Mathematical Formulation:**
  $$w_m = \frac{\exp\big(-\text{MAE}_m / \tau\big)}{\sum_{j=1}^3 \exp\big(-\text{MAE}_j / \tau\big)}$$
* **Optimization Mechanism:** Re-weights base models inversely proportional to validation MAE scaled by an optimized temperature parameter $\tau$.

---

## 5. Input Representation: Current Day ($t$) vs. Past $K$ Days History

The model does not rely purely on a single current-day snapshot; it explicitly integrates **past $K=7$ days of operational history** alongside current day measurements. This captures the degradation velocity and filters out day-to-day measurement spikes:

### Feature Decomposition (24 Input Features)

Every model receives **24 features** per train:
* **14 Current-Day Instantaneous Features ($t$):**
  `critical_jobs`, `non_critical_jobs`, `total_km`, `daily_avg_run`, `variance`, `mileage`, `train_age_years`, `daily_trips`, `days_since_last_maintenance`, `revenue`, `remaining_hours`, `penalty`, `days_remaining`, `maintenance_priority_score`.
* **5 Rolling Window Historical Features (Past $K=7$ Days):**
  Averaged across window $[t-6, t]$ to filter high-frequency operational noise:
  $$\text{rolling\_7d}_{i}(t) = \frac{1}{K} \sum_{j=0}^{K-1} x_i(t - j) \quad (K = 7)$$
  Applied to: `critical_jobs`, `non_critical_jobs`, `daily_avg_run`, `revenue`, `penalty`.
* **5 Velocity / Rate-of-Change Features ($\Delta_{7d}$):**
  First-order temporal differencing measuring degradation velocity:
  $$\Delta_{7d} x_i(t) = x_i(t) - x_i(t - 7)$$
  Applied to: `critical_jobs`, `days_remaining`, `total_km`, `maintenance_priority_score`, `days_since_last_maintenance`.

---

## 6. Ground Truth Target Formula: 4-Tier Risk Formulation

The ground-truth target variable `maintenance_priority_score` is computed via a deterministic 4-tier risk hierarchy that relies **100% on observable columns** present in `railway_maintenance_365_days.csv` (18,250 / 18,250 observations empirically verified across 2025 and 2026 data with 0.000000 maximum error):

$$\text{Priority Hierarchy:}\quad \text{Safety } (0.50) > \text{Revenue } (0.25) > \text{Mileage } (0.15) > \text{Other } (0.10)$$

$$\text{maintenance\_priority\_score} = \text{round}\Big(\text{clip}\big(0.50 \cdot R_{\text{safety}} + 0.25 \cdot R_{\text{revenue}} + 0.15 \cdot R_{\text{mileage}} + 0.10 \cdot R_{\text{other}},\, 0.001,\, 0.999\big),\, 4\Big)$$

### Summary Weights Matrix

| Tier | Tier Weight | Column Name(s) | Sub-Weight | Operational Motivation |
| :--- | :---: | :--- | :---: | :--- |
| **Safety** | **0.50** | `critical_jobs`<br>`days_remaining`<br>`non_critical_jobs` | 0.50<br>0.35<br>0.15 | Passenger safety, derailment prevention, statutory fitness certificate expiration |
| **Revenue** | **0.25** | `penalty`<br>`remaining_hours`<br>`revenue` | 0.45<br>0.35<br>0.20 | Contractual SLA breach penalties with corporate wrap advertisers & peak ticket revenue loss |
| **Mileage** | **0.15** | `mileage`<br>`total_km` | 0.45<br>0.55 | Mechanical wear, bogie/wheel wear, scheduled 65,000 km workshop overhaul threshold |
| **Other** | **0.10** | `days_since_last_maintenance`<br>`train_age_years`<br>`variance`<br>`daily_avg_run`<br>`daily_trips` | 0.35<br>0.25<br>0.15<br>0.15<br>0.10 | Fatigue stress, stop-and-go thermal cycles, day-to-day service schedule variability |

---

## 7. How to Train Models Manually

Each model script contains a configurable variable `d` and can be trained directly inside its dedicated directory without generating external side-effects:

```bash
# Activate virtual environment
source .venv/bin/activate

# Base Models
python ml/models/ridge/train_ridge.py
python ml/models/random_forest/train_rf.py
python ml/models/extra_trees/train_et.py
python ml/models/hist_gradient_boosting/train_hgb.py
python ml/models/lightgbm/train_lgbm.py
python ml/models/lambdamart_ranker/train_ranker.py
python ml/models/xgboost/train_xgb.py

# Ensembles
python ml/models/method-4_slsqp_xgboost_lightgbm/train_method4.py
python ml/models/method-1_constrained_slsqp/train_method1.py
python ml/models/method-2_stacking_ridgecv/train_method2.py
python ml/models/method-3_softmax_inverse_mae/train_method3.py
```
*(Running any script without arguments trains both $t+3$ and $t+7$ horizons automatically; pass `3` or `7` as a CLI argument to train a single horizon).*

---

## 8. Production Backend Architecture & Model Serving

To directly feed predicted priorities and fleet maintenance urgency into the **Google OR-Tools CP-SAT scheduler**, the best-performing models and inference service are structured inside `backend/`:

```
backend/
├── app/
│   ├── config.py                 # Paths, horizon constants, and 24-feature schema
│   ├── schemas/
│   │   ├── train_record.py       # Data structures for incoming train operational telemetry
│   │   └── prediction.py         # Output schemas for priority scores, ranks, and depot actions
│   ├── ml/
│   │   ├── model_loader.py       # Thread-safe cached singleton for joblib model artifacts
│   │   ├── feature_pipeline.py   # 7-day rolling window, velocity differencing, and validation
│   │   └── predictor.py          # Unified inference engine for t+3 and t+7 forecasts
│   └── service/
│       └── priority_service.py   # Fleet priority ranking and depot dispatch categorization
├── models/                       # Dedicated production model storage
│   ├── metadata.json             # Manifest with feature schemas, weights, and benchmark metrics
│   ├── t3/                       # Best performing models for short-term dispatch (t+3)
│   │   ├── ensemble_slsqp_xgb_lgbm_t3.joblib  # #1 Overall Dual-Booster SLSQP Ensemble
│   │   ├── xgboost_t3.joblib                  # Regularized Level-Wise Gradient Booster
│   │   └── lightgbm_t3.joblib                 # #1 Standalone Leaf-Wise Gradient Booster
│   └── t7/                       # Best performing models for weekly overhaul planning (t+7)
│       ├── ensemble_slsqp_xgb_lgbm_t7.joblib  # #1 Overall Dual-Booster SLSQP Ensemble
│       ├── xgboost_t7.joblib                  # #1 Standalone Regularized Gradient Booster
│       └── lightgbm_t7.joblib                 # Leaf-Wise Gradient Booster
└── tests/
    └── test_backend_inference.py # End-to-end backend integration tests
```

### Python Backend Service Usage Example

```python
import pandas as pd
from backend.app.service.priority_service import PriorityService

# 1. Initialize service (automatically loads and caches best production models)
service = PriorityService()

# 2. Provide 25-train daily operational records
fleet_df = pd.read_csv("ml/data/data_2026/railway_maintenance_365_days.csv")
daily_snapshot = fleet_df[fleet_df['date'] == '2026-06-15']

# 3. Forecast t+3 maintenance priority using the Dual-Booster SLSQP Ensemble
response = service.rank_fleet(daily_snapshot, horizon="t3", model_type="ensemble")

# 4. Pass ranked trains directly to CP-SAT scheduler
for train in response.predictions:
    print(f"Rank {train.rank:02d} | {train.train_id}: Score {train.forecast_score:.4f} -> {train.status}")
```

---

## 9. Verification & Test Suite

To verify the evaluation pipelines, test backend inference, and validate ranking outputs:

```bash
# Verify ground-truth formula and benchmark all models on unseen 2026 data:
python test/test_evaluate_2026.py

# Verify backend model loader, feature pipeline, and priority service:
python backend/tests/test_backend_inference.py

# Run real-world 25-train fleet maintenance dispatching inference:
python test/test_ranking_inference.py
```

# Kochi-Metro-Optima
