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
    if not os.path.exists(model_path):
        print("Model not found.")
        return
        
    model = joblib.load(model_path)
    print("Model loaded.")
    
    # Test point: Middle of the Slot (x=60, y=0)
    # This should have a high block probability based on our heatmap.
    test_pt = pd.DataFrame([{
        'x': 60.0, 'y': 0.0,
        'distance': 29.0, 'angle_deg': 0.0,
        'shot_type': 'wrist', 'shooter_role': 'F',
        'is_rush': 0, 'is_rebound': 0, 'period_number': 2,
        'game_state': '5v5', 'relative_game_state': '5v5', 'is_home': 1,
        'score_diff': 0, 'last_event_type': 'faceoff', 'last_event_time_diff': 5.0,
        'dist_from_last_event': 20.0, 'speed_from_last_event': 4.0
    }])
    
    # Prepare
    df_inf = model._prepare_inference_df(test_pt)
    
    # Predict
    p_block = model.model_block.predict_proba(df_inf[model.features_block])[0, 1]
    p_acc = model.model_acc.predict_proba(df_inf[model.features_acc])[0, 1]
    p_fin = model.model_finish.predict_proba(df_inf[model.features_fin])[0, 1]
    
    print(f"\nPredictions for Point (x=60, y=0):")
    print(f"P(Block)   : {p_block:.4f}")
    print(f"P(Accuracy): {p_acc:.4f}")
    print(f"P(Finish)  : {p_fin:.4f}")
    print(f"Final xG   : {(1-p_block)*p_acc*p_fin:.4f}")

    # check average on random data
    print("\n--- Feature Importance (Block Model) ---")
    booster = model.model_block.get_booster()
    imp = booster.get_score(importance_type='gain')
    sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)
    for f, v in sorted_imp[:10]:
        print(f"{f:25}: {v:.1f}")

if __name__ == "__main__":
    diagnose()
