import sys
import os
import pandas as pd
import joblib

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from puck import analyze, data_pipeline

def check_calibration(season):
    print(f"Checking calibration for {season}...")
    try:
        csv_path = analyze.locate_season_csv(season)
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading {season}: {e}")
        return

    # Filter to 5v5 for comparison if appropriate, or keep ALL
    df = data_pipeline.preprocess_features(df, apply_filtering=True)
    
    models = {
        'xgboost_nested': 'analysis/xgs/xg_model_xgboost_nested_20202021.joblib',
        'xgboost_non_nested': 'analysis/xgs/xg_model_xgboost_non_nested_20202021.joblib',
        'nested_xg': 'analysis/xgs/xg_model_nested_tensor_20202021.joblib',
        'non_nested_xg': 'analysis/xgs/xg_model_non_nested_tensor_20202021.joblib',
    }
    
    actual_goals = (df['event'].str.lower() == 'goal').sum()
    total_shots = len(df)
    
    print(f"Actual Goals: {actual_goals}")
    print(f"Total Shots: {total_shots}")
    # print(f"Actual Goal Rate: {actual_goals / total_shots:.4f}" if total_shots > 0 else "0")
    
    mask = df['event'].isin(['shot-on-goal', 'missed-shot', 'goal'])
    
    for name, path in models.items():
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                # Some models might need specific feature formatting
                # predict_proba for GLM might work differently than XGBoost
                # but PredictiveEvaluator uses standard predict_proba(df)[:, 1]
                preds = model.predict_proba(df[mask])[:, 1]
                total_xg = preds.sum()
                print(f"Model {name} Total xG: {total_xg:.2f} (Ratio: {total_xg / actual_goals:.4f})")
            except Exception as e:
                print(f"Error running model {name}: {e}")
        else:
            print(f"Model {name} not found at {path}")
    print("-" * 20)

if __name__ == "__main__":
    check_calibration('20202021')
    check_calibration('20232024')
