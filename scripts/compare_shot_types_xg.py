
import pandas as pd
import numpy as np
import os
import sys
import joblib

# Add project root to path
sys.path.append(os.getcwd())

from puck import data_pipeline, fit_xgboost_nested

def compare_shot_types():
    print("--- Comparing Wrist vs Slap Shot Danger (Calibrated Model) ---")
    
    model_path = os.path.join('analysis', 'xgs', 'xg_model_xgboost_nested_20202021.joblib')
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found.")
        return

    print(f"Loading model: {model_path}")
    clf = joblib.load(model_path)
    
    # Create baseline features
    scenarios = [
        {'name': 'Low Slot (15ft)', 'distance': 15.0, 'angle_deg': 0.0},
        {'name': 'High Slot (30ft)', 'distance': 30.0, 'angle_deg': 0.0},
        {'name': 'Point (55ft)', 'distance': 55.0, 'angle_deg': 10.0},
    ]
    
    results = []
    
    for s in scenarios:
        for st in ['wrist', 'slap']:
            # Create a minimal row
            data = {
                'distance': s['distance'],
                'angle_deg': s['angle_deg'],
                'shot_type': st,
                'x': 89.0 - s['distance'], # Roughly
                'y': 0.0,
                'is_home': 1,
                'relative_game_state': '5v5',
                'period_number': 1,
                'time_elapsed_in_period_s': 300,
                'total_time_elapsed_s': 300,
                'score_diff': 0,
                'shoots_catches': 'L',
                'shooter_role': 'F',
                'is_rebound': 0,
                'is_rush': 0,
                'rebound_angle_change': 0.0,
                'rebound_time_diff': 0.0,
                'last_event_type': 'faceoff',
                'last_event_time_diff': 10.0,
                'dist_from_last_event': 20.0,
                'speed_from_last_event': 2.0,
                'angle_change_last_event': 0.0
            }
            df = pd.DataFrame([data])
            
            # Preprocess to get correct dtypes/categories
            df_proc = clf._prepare_inference_df(df)
            
            # Predict
            prob = clf.predict_proba(df_proc)[0, 1]
            p_block = clf.predict_proba_layer(df_proc, 'block')[0]
            p_acc = clf.predict_proba_layer(df_proc, 'accuracy')[0]
            p_fin = clf.predict_proba_layer(df_proc, 'finish')[0]
            
            results.append({
                'Scenario': s['name'],
                'Type': st,
                'xG': prob,
                'P(Block)': p_block,
                'P(OnNet)': p_acc,
                'P(Goal|Net)': p_fin
            })
            
    res_df = pd.DataFrame(results)
    print("\nComparison Table:")
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    compare_shot_types()
