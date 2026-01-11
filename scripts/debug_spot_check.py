
import os
import sys
import pandas as pd
import numpy as np
import joblib

sys.path.append(os.getcwd())
try:
    from puck import analyze, features, rink
except ImportError:
    print("Could not import puck modules. Make sure you are in the project root.")

def main():
    print("--- Spot Check Analysis: (80, 1) ---")
    
    # 1. Load Model
    model_path = 'analysis/xgs/xg_model_nested_all.joblib'
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    clf = joblib.load(model_path)
    print(f"Loaded {type(clf).__name__}")

    # 2. Construct Data Point
    # x=80, y=1 (Right side of slot if facing goal? No, y=1 is left of center if facing net? 
    # y is width. 0 is center. 1 is slightly off center.)
    
    # Calculate Features exactly as model expects
    x, y = 80.0, 1.0
    goal_x = 89.0
    
    dx = x - goal_x
    dy = y - 0  # goal_y
    distance = np.sqrt(dx**2 + dy**2)
    
    # Angle
    # Vector from goal to shot: (x-goal_x, y-0) = (-9, 1)
    # Ref vector (goalie left): (0, -1)
    # But wait, analyze_model_spatial_weights used:
    # rx, ry = 0.0, -1.0
    # cross = rx * dy - ry * dx
    # dot = rx * dx + ry * dy
    # This matches.
    rx, ry = 0.0, -1.0
    cross = rx * dy - ry * dx
    dot = rx * dx + ry * dy
    angle_rad = np.arctan2(cross, dot)
    angle_deg = (-np.degrees(angle_rad)) % 360.0
    
    print(f"\nFeatures for x={x}, y={y}:")
    print(f"  Distance: {distance:.4f}")
    print(f"  Angle:    {angle_deg:.4f}")

    # Construct DF
    row = {
        'x': x,
        'y': y,
        'distance': distance,
        'angle_deg': angle_deg,
        'game_state': '5v5',
        'shot_type': 'wrist',
        'is_rebound': 0,
        'rebound_angle_change': 0,
        'rebound_time_diff': 0,
        'is_rush': 0,
        'is_net_empty': 0, # CRITICAL: Ensure goalie is in net
        'last_event_type': 'facetoff',
        'last_event_time_diff': 10,
        'shooter_role': 'F',
        'shoots_catches': 'L',
        'score_diff': 0,
        'period_number': 1,
        'time_elapsed_in_period_s': 600,
        'total_time_elapsed_s': 600,
        'event': 'shot-on-goal'
    }
    df = pd.DataFrame([row])
    
    # Predict
    prob = clf.predict_proba(df)[0, 1]
    print(f"\nModel Prediction (5v5 Wrist): {prob:.4%}")
    
    # 3. Empirical Check
    data_file = 'data/20232024/20232024_df.csv' 
    if os.path.exists(data_file):
        print(f"\nChecking empirical data from {data_file}...")
        df_real = pd.read_csv(data_file)
        
        # Filter for similar location (+/- 2 ft)
        # Note: 'shot_type' is likely lower case 'wrist' based on training code, but check both
        mask = (df_real['x'].between(78, 82)) & (df_real['y'].between(-1, 3)) & (df_real['game_state'] == '5v5')
        
        if 'shot_type' in df_real.columns:
            mask = mask & df_real['shot_type'].astype(str).str.lower().eq('wrist')
            
        # Also exclude empty net if poss
        if 'is_net_empty' in df_real.columns:
             mask = mask & (df_real['is_net_empty'] == 0)
        
        subset = df_real[mask]
        print(f"Found {len(subset)} shots in region (x=78-82, y=-1 to 3, 5v5 wrist).")
        if len(subset) > 0:
            goals = subset['event'].apply(lambda e: 1 if str(e).lower() in ['goal'] else 0).sum()
            print(f"Actual Goals: {goals}")
            print(f"Empirical Rate: {goals/len(subset):.4%}")
            
            print("\nSample shots:")
            print(subset[['x', 'y', 'event', 'shot_type']].head())
    else:
        print(f"Data file {data_file} not found.")

if __name__ == "__main__":
    main()
