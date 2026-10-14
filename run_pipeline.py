import sys
import subprocess
import argparse
from pathlib import Path
import joblib

from src.train import run_training_pipeline
from src.predict import predict_validation, predict_december_chart_inputs
from src.config import BASE_DIR, OUTPUT_PREDICTIONS_PATH, OUTPUT_DECEMBER_PATH


def main():
    parser = argparse.ArgumentParser(description="Run Spotter Freight Rate ML Pipeline.")
    parser.add_argument("--retrain", action="store_true", help="Force model retraining.")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("      SPOTTER MACHINE LEARNING ENGINEER ASSESSMENT PIPELINE")
    print("="*70)

    models_dir = BASE_DIR / "models"
    models_exist = (
        (models_dir / "lgb_models.joblib").is_file() and
        (models_dir / "cb_models.joblib").is_file() and
        (models_dir / "equip_meds.joblib").is_file() and
        (models_dir / "mkt_means.joblib").is_file()
    )

    if models_exist and not args.retrain:
        print("Found pre-trained production models in models/. Loading from disk...")
        lgb_models = joblib.load(models_dir / "lgb_models.joblib")
        cb_models = joblib.load(models_dir / "cb_models.joblib")
        equip_meds = joblib.load(models_dir / "equip_meds.joblib")
        mkt_means = joblib.load(models_dir / "mkt_means.joblib")
    else:
        # 1. Train Models (Expanding Window CV + 5-Fold Cross-Validation Ensemble)
        lgb_models, cb_models, equip_meds, mkt_means, _ = run_training_pipeline()

    # 2. Predict validation.csv
    predict_validation(lgb_models, cb_models, equip_meds, mkt_means)

    # 3. Predict december_chart_inputs.csv
    predict_december_chart_inputs(lgb_models, cb_models, equip_meds)

    # 4. Run official score.py
    print("\n" + "="*70)
    print("STEP 5: RUNNING OFFICIAL SCORER (score.py)")
    print("="*70)

    score_cmd = [
        sys.executable,
        str(BASE_DIR / "score.py"),
        "--predictions", str(OUTPUT_PREDICTIONS_PATH),
        "--december-predictions", str(OUTPUT_DECEMBER_PATH),
        "--output-dir", str(BASE_DIR / "scorer_results")
    ]
    
    res = subprocess.run(score_cmd, capture_output=True, text=True)
    print("Scorer Return Code:", res.returncode)
    print("Scorer STDOUT:\n", res.stdout)
    if res.stderr:
        print("Scorer STDERR:\n", res.stderr)

    if res.returncode != 0:
        print("ERROR: score.py failed validation!")
        sys.exit(res.returncode)

    # 5. Generate Technical Reports (DOCX and PDF)
    print("\n" + "="*70)
    print("STEP 6: GENERATING TECHNICAL ASSESSMENT REPORTS (DOCX & PDF)")
    print("="*70)
    from src.report_generator import generate_reports
    generate_reports()

    print("\nSUCCESS: All predictions validated, chart and reports generated successfully!")


if __name__ == "__main__":
    main()
