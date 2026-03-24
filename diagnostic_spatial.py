import joblib
import pandas as pd
import numpy as np
from puck import data_pipeline, rink

def diagnostic():
    models = {
        "Nested GLM": "analysis/xgs/xg_model_nested_tensor_20202021.joblib",
        "Non-Nested GLM": "analysis/xgs/xg_model_non_nested_tensor_20202021.joblib",
        "Nested XGBoost": "analysis/xgs/xg_model_xgboost_nested_20202021.joblib",
        "Non-Nested XGBoost": "analysis/xgs/xg_model_xgboost_non_nested_20202021.joblib"
    }
    
    # Test points (standard Right-Attack orientation, Goal at 89, 0)
    test_points = [
        {"x": 89.0, "y": 0.0, "name": "Direct Front (Goal)"},
        {"x": 70.0, "y": 0.0, "name": "Center Ice (20ft out)"},
        {"x": 70.0, "y": 20.0, "name": "Right Wing (20x20)"},
        {"x": 70.0, "y": -20.0, "name": "Left Wing (20x20)"},
        {"x": 25.0, "y": 0.0, "name": "Blue Line (64ft out)"}
    ]
    
    df_raw = pd.DataFrame(test_points)
    df_raw['event'] = 'shot-on-goal'
    df_raw['shot_type'] = 'wrist'
    df_raw['shooter_role'] = 'F'
    df_raw['shoots_catches'] = 'L'
    df_raw['relative_game_state'] = '5v5'
    df_raw['game_state'] = '5v5'
    df_raw['is_home'] = 1
    df_raw['is_rush'] = 0
    df_raw['is_rebound'] = 0
    df_raw['last_event_type'] = 'faceoff'
    df_raw['period_number'] = 2
    
    # Add ALL possible features but ensure float/correct types
    df_raw['angle_change_last_event'] = 0.0
    df_raw['speed_from_last_event'] = 0.0
    df_raw['dist_from_last_event'] = 0.0
    df_raw['last_event_time_diff'] = 5.0
    df_raw['time_elapsed_in_period_s'] = 600.0
    df_raw['total_time_elapsed_s'] = 600.0
    df_raw['score_diff'] = 0.0
    df_raw['is_net_empty'] = 0.0
    df_raw['rebound_angle_change'] = 0.0
    df_raw['rebound_time_diff'] = 0.0
    df_raw['is_home'] = df_raw['is_home'].astype(float)
    df_raw['is_rush'] = df_raw['is_rush'].astype(float)
    df_raw['is_rebound'] = df_raw['is_rebound'].astype(float)
    
    # Preprocess using Pipeline Logic (which is now 90-center)
    df_pre = data_pipeline.preprocess_features(df_raw, is_training=False)
    
    print("--- PIEPLINE PREPROCESSED FEATURES ---")
    print(df_pre[['x', 'y', 'distance', 'angle_deg']])
    
    print("\n--- MODEL PREDICTIONS ---")
    for name, path in models.items():
        try:
            model = joblib.load(path)
            # Ensure we pass the preprocessed columns in the right order
            probs = model.predict_proba(df_pre)[:, 1]
            print(f"\n{name}:")
            for i, p in enumerate(probs):
                print(f"  {test_points[i]['name']}: {p:.4f}")
        except Exception as e:
            print(f"Error loading {name}: {e}")

if __name__ == "__main__":
    diagnostic()
