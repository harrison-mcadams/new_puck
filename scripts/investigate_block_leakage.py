
import sys, os, joblib
import pandas as pd
import numpy as np

sys.path.append(os.getcwd())
from puck import features as feature_util, data_pipeline

def investigate():
    model_path = 'analysis/xgs/xg_model_nested.joblib'
    model = joblib.load(model_path)
    
    # Base Case: High Slot Forward Shot (65ft)
    base_df = pd.DataFrame({
        'distance': [65.0],
        'angle_deg': [0.0],
        'shooter_role': ['F'],
        'event': ['shot-on-goal'],
        'game_state': ['5v5'],
        'shot_type': ['Wrist Shot'],
        'is_rebound': [0],
        'is_rush': [0],
        'period_number': [2],
        'score_diff': [0],
        'last_event_type': ['Faceoff'],
        'last_event_time_diff': [10.0],
        'time_elapsed_in_period_s': [600.0],
        'total_time_elapsed_s': [1800.0],
        'shoots_catches': ['L']
    })
    
    # We must use _prepare_df or similar to ensure features match
    def get_p(df):
        return model.predict_proba_layer(df, layer='block')[0]

    print(f"Base P(Blocked) for F at 65ft: {get_p(base_df):.4f}")
    
    # Ablation
    # 1. Change Role
    df_d = base_df.copy()
    df_d['shooter_role'] = 'D'
    print(f"If Role='D': {get_p(df_d):.4f}")
    
    # 2. Change Angle
    df_angle = base_df.copy()
    df_angle['angle_deg'] = 30.0
    print(f"If Angle=30: {get_p(df_angle):.4f}")
    
    # 3. Change Last Event
    events = ['Faceoff', 'Hit', 'Giveaway', 'Shot-on-Goal', 'Blocked-Shot', 'Stoppage']
    print("\nEffect of last_event_type:")
    for e in events:
        tmp = base_df.copy()
        tmp['last_event_type'] = e
        print(f"  {e:15}: {get_p(tmp):.4f}")
        
    # 4. Change Score Diff
    print("\nEffect of score_diff:")
    for s in [-2, 0, 2]:
        tmp = base_df.copy()
        tmp['score_diff'] = s
        print(f"  {s:15}: {get_p(tmp):.4f}")

    # 5. Check Categorical Leakage (Unknowns)
    print("\nStrategic Neutralization:")
    neutral_df = base_df.copy()
    neutral_df['last_event_type'] = 'Unknown'
    neutral_df['game_state'] = '5v5'
    neutral_df['shoots_catches'] = 'Unknown'
    print(f"  All Categoricals 'Unknown': {get_p(neutral_df):.4f}")

if __name__ == "__main__":
    investigate()
