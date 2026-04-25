import sys
import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from puck import fit_xgboost_alternate, config

def diagnose():
    model_path = 'analysis/xgs/xg_model_xgboost_alternate_modern_era.joblib'
    model = joblib.load(model_path)
    
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
    
    print("\n--- Feature Values for Point (x=60, y=0) ---")
    feats = model.features_block
    for f in feats:
        val = df_inf[f].iloc[0]
        if isinstance(val, (float, np.float32, np.float64)):
             if val > 0.0001:
                print(f"{f:25}: {val:.4f}")
        else:
            print(f"{f:25}: {val}")

    p_block = model.model_block.predict_proba(df_inf[model.features_block])[0, 1]
    print(f"\nFinal P(Block): {p_block:.6f}")

if __name__ == "__main__":
    diagnose()
