import sys
import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from puck import fit_xgboost_alternate

def diagnose():
    model_path = 'analysis/xgs/xg_model_xgboost_alternate_modern_era.joblib'
    model = joblib.load(model_path)
    
    # 1. Get raw training data for comparison
    from puck import analyze, data_pipeline
    df = data_pipeline.preprocess_features(pd.read_csv(analyze.locate_season_csv('20232024')), apply_filtering=True)
    
    # Test point
    test_pt = pd.DataFrame([{
        'x': 60.0, 'y': 0.0,
        'distance': 29.0, 'angle_deg': 0.0,
        'shot_type': 'wrist', 'shooter_role': 'F',
        'is_rush': 0, 'is_rebound': 0, 'period_number': 2,
        'game_state': '5v5', 'relative_game_state': '5v5', 'is_home': 1,
        'score_diff': 0, 'last_event_type': 'faceoff', 'last_event_time_diff': 5.0,
        'dist_from_last_event': 20.0, 'speed_from_last_event': 4.0
    }])
    
    df_inf = model._prepare_inference_df(test_pt)
    
    # 2. Check internal booster
    booster = model.model_block.get_booster()
    
    # Create DMatrix
    X = df_inf[model.features_block]
    # Ensure it's category
    for col in X.columns:
        if isinstance(X[col].dtype, pd.CategoricalDtype):
             pass # already good
             
    dmat = xgb.DMatrix(X, enable_categorical=True)
    
    raw_p = booster.predict(dmat)
    print(f"Raw Booster Prediction: {raw_p[0]:.6f}")
    
    # 3. Compare with some actual rows from training data
    df_train_inf = model._prepare_inference_df(df.head(100))
    train_p = model.model_block.predict_proba(df_train_inf[model.features_block])[:, 1]
    print(f"Mean Prediction on 100 train rows: {train_p.mean():.4f}")
    
    # Check if ANY of them are > 0.1
    print(f"Max Prediction on 100 train rows:  {train_p.max():.4f}")

if __name__ == "__main__":
    diagnose()
