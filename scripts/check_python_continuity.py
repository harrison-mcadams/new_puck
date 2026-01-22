import joblib
import pandas as pd
import numpy as np
import sys
import os

# Ensure puck package is importable
sys.path.append(os.getcwd())

from puck.fit_glm_nested import NestedGLM, TensorSpline

def main():
    model_path = "analysis/xgs/xg_model_nested.joblib"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    print(f"Loading model from {model_path}...")
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Create two test points slightly offset from 0/360 boundary
    # Fix other params to defaults
    row_base = {
        'period_number': 2,
        'time_elapsed_in_period_s': 600,
        'total_time_elapsed_s': 1800,
        'last_event_time_diff': 10,
        'home_team_defending_side': 'left',
        'shooter_role': 'F',
        'game_state': '5v5',
        'shot_type': 'wrist',
        'shoots_catches': 'L',
        'is_rush': 0,
        'is_rebound': 0,
        'last_event_type': 'faceoff',
        'score_diff': 0,
        'speed_from_last_event': 0,
        'dist_from_last_event': 0,
        'rebound_angle_change': 0,
        'rebound_time_diff': 0,
        'angle_change_last_event': 0,
        'dist_from_last_event': 0,
        'dist_angle': 0,
    }
    
    # Test Point 1: Near 0
    r1 = row_base.copy()
    r1['distance'] = 39.0
    r1['angle_deg'] = 0.001
    
    # Test Point 2: Near 360
    r2 = row_base.copy()
    r2['distance'] = 39.0
    r2['angle_deg'] = 359.999
    
    df = pd.DataFrame([r1, r2])
    
    print("Predicting probabilities...")
    try:
        # returns [P(Unblocked), P(Blocked)]? No, usually [P(not_goal), P(goal)] for predict_proba
        # But this is NestedGLM. predict_proba returns [P(no_goal), P(goal)]
        probs = model.predict_proba(df)
        
        p1 = probs[0, 1]
        p2 = probs[1, 1]
        
        print(f"P(Goal) at 0.001 deg: {p1:.6f}")
        print(f"P(Goal) at 359.999 deg: {p2:.6f}")
        
        diff = abs(p1 - p2)
        print(f"Difference: {diff:.6f}")
        
        if diff < 1e-4:
            print("CONCLUSION: Model is CONTINUOUS at angular boundary.")
        else:
            print("CONCLUSION: Model is DISCONTINUOUS at angular boundary (Cliff confirmed).")
            
        # Also check internal layers if possible
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
