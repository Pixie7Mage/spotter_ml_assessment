import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import KFold
import lightgbm as lgb
from catboost import CatBoostRegressor
import joblib

from .config import (
    TRAIN_DATA_PATH,
    CITY_COORDINATES,
    RANDOM_SEED,
    BASE_DIR
)
from .data_cleaning import preprocess_data
from .feature_engineering import engineer_features

# Shift-robust feature specification:
# NO discrete city IDs (avoids breakages on unseen evaluation cities)
# NO dayofyear / month (avoids tree extrapolation errors on Nov-Dec)
NUMERIC_FEATURES = [
    "distance", "log_dist", "inv_dist",
    "clean_weight", "log_weight", "ton_miles", "weight_per_mile", "is_heavy", "is_missing_weight",
    "clean_mkt", "log_mkt", "is_missing_mkt",
    "quote_signal", "base_quote", "mkt_adj_quote", "rate_per_ton_mile_signal",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "haversine", "circuity", "lat_diff", "lon_diff", "bearing",
    "dayofweek", "is_weekend", "dow_sin", "dow_cos"
]
CATEGORICAL_FEATURES = ["equipment"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes comprehensive regression metrics including WAPE."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100.0)
    wape = float(np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100.0)
    r2 = float(r2_score(y_true, y_pred))
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "WAPE": wape,
        "R2": r2,
    }


def train_expanding_window_validation(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluates models across Expanding-Window Time CV folds:
    - Fold 1: Train Jan-Apr -> Val May-Jun (2 months)
    - Fold 2: Train Jan-Jun -> Val Jul-Aug (2 months)
    - Fold 3: Train Jan-Aug -> Val Sep-Oct (2 months - Final Out-of-Time Holdout)
    Target is log(ratio) = log(posted_rate / (distance * quote_signal)).
    """
    print("\n" + "="*60)
    print("STEP 1: EXPANDING-WINDOW TIME CROSS-VALIDATION")
    print("="*60)

    windows = [
        (df["month"] <= 4, (df["month"] >= 5) & (df["month"] <= 6), "Window 1: Train Jan-Apr -> Val May-Jun"),
        (df["month"] <= 6, (df["month"] >= 7) & (df["month"] <= 8), "Window 2: Train Jan-Jun -> Val Jul-Aug"),
        (df["month"] <= 8, (df["month"] >= 9) & (df["month"] <= 10), "Window 3: Train Jan-Aug -> Val Sep-Oct (Final Holdout)")
    ]

    window_results = []
    final_holdout_models = {}

    for tr_mask, val_mask, window_name in windows:
        X_tr = df.loc[tr_mask, ALL_FEATURES].copy()
        y_tr_log_ratio = df.loc[tr_mask, "log_ratio"].values
        
        X_val = df.loc[val_mask, ALL_FEATURES].copy()
        y_val_rate = df.loc[val_mask, "posted_rate"].values
        base_val = df.loc[val_mask, "base_quote"].values

        # Baseline: distance * quote_signal
        base_metrics = compute_metrics(y_val_rate, base_val)

        # LightGBM setup
        X_tr_lgb = X_tr.copy()
        X_val_lgb = X_val.copy()
        X_tr_lgb["equipment"] = X_tr_lgb["equipment"].astype("category")
        X_val_lgb["equipment"] = X_val_lgb["equipment"].astype("category")

        m_lgb = lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=1000,
            learning_rate=0.02,
            num_leaves=35,
            min_child_samples=30,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=RANDOM_SEED,
            verbose=-1,
            n_jobs=-1
        )
        m_lgb.fit(X_tr_lgb, y_tr_log_ratio)
        pred_lgb = base_val * np.exp(m_lgb.predict(X_val_lgb))
        lgb_metrics = compute_metrics(y_val_rate, pred_lgb)

        # CatBoost setup
        X_tr_cb = X_tr.copy()
        X_val_cb = X_val.copy()
        X_tr_cb["equipment"] = X_tr_cb["equipment"].astype(str)
        X_val_cb["equipment"] = X_val_cb["equipment"].astype(str)

        m_cb = CatBoostRegressor(
            loss_function="MAE",
            iterations=1000,
            learning_rate=0.03,
            depth=6,
            cat_features=["equipment"],
            random_seed=RANDOM_SEED,
            verbose=0,
            thread_count=-1
        )
        m_cb.fit(X_tr_cb, y_tr_log_ratio)
        pred_cb = base_val * np.exp(m_cb.predict(X_val_cb))
        cb_metrics = compute_metrics(y_val_rate, pred_cb)

        # Blend (80% LightGBM + 20% CatBoost)
        pred_blend = 0.80 * pred_lgb + 0.20 * pred_cb
        blend_metrics = compute_metrics(y_val_rate, pred_blend)

        print(f"\n--- {window_name} ---")
        print(f"Train: {tr_mask.sum():,} loads | Val: {val_mask.sum():,} loads")
        print(f"Baseline MAE: ${base_metrics['MAE']:.2f}")
        print(f"Ensemble MAE: ${blend_metrics['MAE']:.2f} ({(1 - blend_metrics['MAE']/base_metrics['MAE'])*100:.1f}% reduction)")
        print(f"Ensemble MAPE: {blend_metrics['MAPE']:.2f}% | WAPE: {blend_metrics['WAPE']:.2f}% | R2: {blend_metrics['R2']:.4f}")

        res_dict = {
            "window": window_name,
            "train_size": int(tr_mask.sum()),
            "val_size": int(val_mask.sum()),
            "Baseline": base_metrics,
            "LightGBM": lgb_metrics,
            "CatBoost": cb_metrics,
            "Ensemble": blend_metrics
        }
        window_results.append(res_dict)

        if "Window 3" in window_name:
            final_holdout_models = {
                "Baseline": base_metrics,
                "LightGBM": lgb_metrics,
                "CatBoost": cb_metrics,
                "Ensemble": blend_metrics
            }

    mean_mae = float(np.mean([w["Ensemble"]["MAE"] for w in window_results]))
    mean_mape = float(np.mean([w["Ensemble"]["MAPE"] for w in window_results]))
    mean_wape = float(np.mean([w["Ensemble"]["WAPE"] for w in window_results]))
    mean_r2 = float(np.mean([w["Ensemble"]["R2"] for w in window_results]))

    print(f"\n=== Expanding Window Average Performance ===")
    print(f"Mean MAE:  ${mean_mae:.2f}")
    print(f"Mean MAPE: {mean_mape:.2f}%")
    print(f"Mean WAPE: {mean_wape:.2f}%")
    print(f"Mean R2:   {mean_r2:.4f}")

    return {
        "windows": window_results,
        "final_out_of_time_holdout": final_holdout_models,
        "expanding_window_mean": {
            "MAE": mean_mae,
            "MAPE": mean_mape,
            "WAPE": mean_wape,
            "R2": mean_r2
        }
    }


def train_production_kfold_models(
    df: pd.DataFrame, n_splits: int = 5
) -> Tuple[List[Any], List[Any], Dict[str, float]]:
    """
    Trains a 5-fold cross-validated bagged ensemble across the full 48,000 labeled loads.
    Averaging predictions from the 5 folds minimizes test variance on unseen periods.
    """
    print("\n" + "="*60)
    print(f"STEP 2: 5-FOLD BAGGED PRODUCTION ENSEMBLE TRAINING (N={len(df):,})")
    print("="*60)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    
    oof_preds_lgb = np.zeros(len(df))
    oof_preds_cb = np.zeros(len(df))
    
    lgb_models = []
    cb_models = []

    X = df[ALL_FEATURES].copy()
    y_log_ratio = df["log_ratio"].values
    base_quotes = df["base_quote"].values
    y_true_rates = df["posted_rate"].values

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_log_ratio), 1):
        X_tr, y_tr = X.iloc[train_idx].copy(), y_log_ratio[train_idx]
        X_val, y_val = X.iloc[val_idx].copy(), y_log_ratio[val_idx]
        base_val = base_quotes[val_idx]
        y_val_true = y_true_rates[val_idx]

        # LightGBM
        X_tr_lgb = X_tr.copy()
        X_val_lgb = X_val.copy()
        X_tr_lgb["equipment"] = X_tr_lgb["equipment"].astype("category")
        X_val_lgb["equipment"] = X_val_lgb["equipment"].astype("category")

        model_lgb = lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=1000,
            learning_rate=0.02,
            num_leaves=35,
            min_child_samples=30,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=RANDOM_SEED + fold,
            verbose=-1,
            n_jobs=-1
        )
        model_lgb.fit(X_tr_lgb, y_tr)
        pred_lgb_rate = base_val * np.exp(model_lgb.predict(X_val_lgb))
        oof_preds_lgb[val_idx] = pred_lgb_rate
        lgb_models.append(model_lgb)

        # CatBoost
        X_tr_cb = X_tr.copy()
        X_val_cb = X_val.copy()
        X_tr_cb["equipment"] = X_tr_cb["equipment"].astype(str)
        X_val_cb["equipment"] = X_val_cb["equipment"].astype(str)

        model_cb = CatBoostRegressor(
            loss_function="MAE",
            iterations=1000,
            learning_rate=0.03,
            depth=6,
            cat_features=["equipment"],
            random_seed=RANDOM_SEED + fold,
            verbose=0,
            thread_count=-1
        )
        model_cb.fit(X_tr_cb, y_tr)
        pred_cb_rate = base_val * np.exp(model_cb.predict(X_val_cb))
        oof_preds_cb[val_idx] = pred_cb_rate
        cb_models.append(model_cb)

        fold_mae = mean_absolute_error(y_val_true, 0.80 * pred_lgb_rate + 0.20 * pred_cb_rate)
        print(f"Fold {fold}/{n_splits} - Ensemble MAE: ${fold_mae:.2f}")

    oof_ensemble = 0.80 * oof_preds_lgb + 0.20 * oof_preds_cb
    kfold_metrics = compute_metrics(y_true_rates, oof_ensemble)
    print("\n=== Random K-Fold OOF Metrics (Internal Reference) ===")
    print(f"OOF MAE:  ${kfold_metrics['MAE']:.2f}")
    print(f"OOF MAPE: {kfold_metrics['MAPE']:.2f}% | WAPE: {kfold_metrics['WAPE']:.2f}% | R2: {kfold_metrics['R2']:.4f}")
    print("NOTE: Honest expected forward performance is given by Expanding Window CV ($116-$119 MAE), not K-fold.")

    return lgb_models, cb_models, kfold_metrics


def run_training_pipeline() -> Tuple[List[Any], List[Any], Dict[str, float], Dict[str, float], Dict[str, Any]]:
    """
    Executes end-to-end data preparation, expanding window evaluation, and 5-fold ensemble training.
    """
    raw_train = pd.read_csv(TRAIN_DATA_PATH)
    clean_train, equip_meds, mkt_means = preprocess_data(raw_train, CITY_COORDINATES)
    fe_train = engineer_features(clean_train)

    fe_train["month"] = fe_train["date"].dt.month
    fe_train["ratio"] = fe_train["posted_rate"] / fe_train["base_quote"]
    fe_train["log_ratio"] = np.log(fe_train["ratio"].clip(lower=0.1, upper=20.0))

    # 1. Expanding Window Time Cross-Validation
    expanding_results = train_expanding_window_validation(fe_train)

    # 2. Production 5-Fold Bagged Ensemble
    lgb_models, cb_models, kfold_metrics = train_production_kfold_models(fe_train)

    # Save artifacts
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(lgb_models, models_dir / "lgb_models.joblib")
    joblib.dump(cb_models, models_dir / "cb_models.joblib")
    joblib.dump(equip_meds, models_dir / "equip_meds.joblib")
    joblib.dump(mkt_means, models_dir / "mkt_means.joblib")

    metrics_summary = {
        "temporal_validation": expanding_results["final_out_of_time_holdout"],
        "expanding_window_validation": expanding_results,
        "kfold_validation": kfold_metrics
    }
    with open(models_dir / "metrics_summary.json", "w") as f:
        json.dump(metrics_summary, f, indent=4)

    return lgb_models, cb_models, equip_meds, mkt_means, metrics_summary


if __name__ == "__main__":
    run_training_pipeline()
