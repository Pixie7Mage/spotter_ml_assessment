import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Any, Dict

from .config import (
    VALIDATION_DATA_PATH,
    DECEMBER_INPUTS_PATH,
    OUTPUT_PREDICTIONS_PATH,
    OUTPUT_DECEMBER_PATH,
    CITY_COORDINATES,
    BASE_DIR
)
from .data_cleaning import preprocess_data
from .feature_engineering import engineer_features
from .train import ALL_FEATURES, CATEGORICAL_FEATURES


def predict_validation(
    lgb_models: List[Any],
    cb_models: List[Any],
    equip_meds: Dict[str, float],
    mkt_means: Dict[str, float]
) -> pd.DataFrame:
    """
    Generates rate predictions for all 12,000 loads in validation.csv.
    Uses log(ratio) ensemble to enforce strictly positive, scaling predictions.
    """
    print("\n" + "="*60)
    print("STEP 3: GENERATING PREDICTIONS FOR VALIDATION.CSV (12,000 ROWS)")
    print("="*60)

    val_df = pd.read_csv(VALIDATION_DATA_PATH)
    clean_val, _, _ = preprocess_data(val_df, CITY_COORDINATES, equip_meds, mkt_means)
    fe_val = engineer_features(clean_val)

    X_val = fe_val[ALL_FEATURES].copy()
    base_quotes = fe_val["base_quote"].values

    # LightGBM predictions on log_ratio
    X_val_lgb = X_val.copy()
    for c in CATEGORICAL_FEATURES:
        X_val_lgb[c] = X_val_lgb[c].astype("category")

    lgb_log_r = np.zeros(len(val_df))
    for m in lgb_models:
        lgb_log_r += m.predict(X_val_lgb) / len(lgb_models)

    # CatBoost predictions on log_ratio
    cb_X = X_val.copy()
    for c in CATEGORICAL_FEATURES:
        cb_X[c] = cb_X[c].astype(str)

    cb_log_r = np.zeros(len(val_df))
    for m in cb_models:
        cb_log_r += m.predict(cb_X) / len(cb_models)

    # Ensemble blend on log_ratio (80% LightGBM + 20% CatBoost)
    ensemble_log_r = 0.80 * lgb_log_r + 0.20 * cb_log_r
    final_predictions = base_quotes * np.exp(ensemble_log_r)
    final_predictions = np.clip(final_predictions, a_min=50.0, a_max=None)

    output_df = pd.DataFrame({
        "load_id": val_df["load_id"],
        "predicted_rate": np.round(final_predictions, 2)
    })

    output_df.to_csv(OUTPUT_PREDICTIONS_PATH, index=False)
    print(f"Saved final validation predictions to: {OUTPUT_PREDICTIONS_PATH}")
    print(output_df.head(5))
    return output_df


def predict_december_chart_inputs(
    lgb_models: List[Any],
    cb_models: List[Any],
    equip_meds: Dict[str, float],
    val_df_path: Path = VALIDATION_DATA_PATH
) -> pd.DataFrame:
    """
    Generates rate predictions for the 31 fixed-input December loads.
    Enriched with observed December market index and historical lane baseline quote signal.
    """
    print("\n" + "="*60)
    print("STEP 4: GENERATING DECEMBER 2025 FIXED INPUT PREDICTIONS (31 ROWS)")
    print("="*60)

    dec_df = pd.read_csv(DECEMBER_INPUTS_PATH)
    val_df = pd.read_csv(val_df_path)

    # Daily mean market index from validation.csv for December dates
    val_df["date_str"] = pd.to_datetime(val_df["date"]).dt.strftime("%Y-%m-%d")
    dec_daily_mkt = val_df.groupby("date_str")["market_index"].mean().to_dict()

    dec_df["date"] = pd.to_datetime(dec_df["date"])
    date_str = dec_df["date"].dt.strftime("%Y-%m-%d")

    # Supply domain market indicators
    dec_df["clean_mkt"] = date_str.map(dec_daily_mkt).fillna(1.0)
    dec_df["quote_signal"] = 2.00338  # Historical Dry Van Lexington->Fort Wayne quote signal mean
    dec_df["clean_weight"] = dec_df["weight"].abs().fillna(32000.0)

    # Fill coordinates for Lexington -> Fort Wayne
    dec_df = preprocess_data(dec_df, CITY_COORDINATES, equip_meds)[0]
    fe_dec = engineer_features(dec_df)

    X_dec = fe_dec[ALL_FEATURES].copy()
    base_quotes = fe_dec["base_quote"].values

    # LightGBM
    X_dec_lgb = X_dec.copy()
    for c in CATEGORICAL_FEATURES:
        X_dec_lgb[c] = X_dec_lgb[c].astype("category")

    lgb_log_r = np.zeros(len(dec_df))
    for m in lgb_models:
        lgb_log_r += m.predict(X_dec_lgb) / len(lgb_models)

    # CatBoost
    cb_X = X_dec.copy()
    for c in CATEGORICAL_FEATURES:
        cb_X[c] = cb_X[c].astype(str)

    cb_log_r = np.zeros(len(dec_df))
    for m in cb_models:
        cb_log_r += m.predict(cb_X) / len(cb_models)

    ensemble_log_r = 0.80 * lgb_log_r + 0.20 * cb_log_r
    dec_preds = base_quotes * np.exp(ensemble_log_r)
    dec_preds = np.clip(dec_preds, a_min=50.0, a_max=None)

    dec_output = pd.read_csv(DECEMBER_INPUTS_PATH)
    dec_output["predicted_rate"] = np.round(dec_preds, 2)

    dec_output.to_csv(DECEMBER_INPUTS_PATH, index=False)
    dec_output.to_csv(OUTPUT_DECEMBER_PATH, index=False)

    print(f"Saved December predictions to: {OUTPUT_DECEMBER_PATH}")
    print(dec_output.head(5))
    return dec_output
