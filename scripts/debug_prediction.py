
import sys
import os
import pandas as pd
import joblib
import xgboost as xgb
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from puck import fit_xgboost_nested

def main():
    model_path = 'analysis/xgs/xg_model_nested.joblib'
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    print(f"Model type: {type(model)}")
    print(f"Block model type: {type(model.model_block)}")
    
    # Create dummy data
    df = pd.DataFrame({
        'distance': [30.0],
        'angle_deg': [15.0],
        'game_state': ['5v5'],
        'shot_type': ['wrist'],
        'is_rebound': [0],
        'is_rush': [0],
        'shoots_catches': ['L'],
        'last_event_type': ['Faceoff'],
        'last_event_time_diff': [10.0],
        'rebound_angle_change': [0.0],
        'rebound_time_diff': [0.0],
        'time_elapsed_in_period_s': [600.0],
        'score_diff': [0],
        'period_number': [1],
        'total_time_elapsed_s': [600.0]
    })
    
    # Ensure all features exist
    for f in model.features:
        if f not in df.columns:
            df[f] = 0
            
    print("Predicting...")
    try:
        probs = model.predict_proba(df)
        print(f"Probabilities: {probs}")
    except Exception as e:
        print(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
