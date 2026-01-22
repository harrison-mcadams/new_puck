
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

def calc_feats(x, y):
    goal_x = 89.0
    dx = x - goal_x
    dy = y
    
    dist = np.sqrt(dx**2 + dy**2)
    
    rx, ry = 0.0, -1.0 
    cross = rx * dy - ry * dx
    dot = rx * dx + ry * dy
    angle_rad_ccw = np.arctan2(cross, dot)
    angle_deg = (-np.degrees(angle_rad_ccw)) % 360.0
    
    return dist, angle_deg

def probe_model():
    model_path = Path('analysis/xgs/xg_model_nested_tensor.joblib')
    print(f"Loading model from {model_path}...")
    try:
        clf = joblib.load(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    x_fixed = 50.0 
    ys = np.linspace(-5, 5, 21)
    
    rows = []
    for y in ys:
        d, a = calc_feats(x_fixed, y)
        rows.append({
            'distance': d,
            'angle_deg': a,
            'game_state': '5v5',
            'score_diff': 0, # Int
            'period_number': 2, # Int
            'time_elapsed_in_period_s': 600.0,
            'total_time_elapsed_s': 1800.0,
            'dist_from_last_event': 0.0,
            'speed_from_last_event': 0.0,
            'is_rush': 0,
            'is_rebound': 0,
            'shooter_role': 'F',
            'shoots_catches': 'L',
            'shot_type': 'wrist',
            'last_event_type': 'faceoff',
            'rebound_angle_change': 0.0,
            'rebound_time_diff': 0.0,
            'last_event_time_diff': 10.0, # Float
            'angle_change_last_event': 0.0, # Added this
            'home_team_defending_side': 'right', # Added this
            'player_name': 'Unknown',
            'team_abbrev': 'UNK', 
            'home_abb': 'BUF', 
            'away_abb': 'BOS'
        })
        
    df = pd.DataFrame(rows)
    
    # Check features
    if hasattr(clf, 'features'):
        missing = [f for f in clf.features if f not in df.columns]
        if missing:
            print(f"MISSING FEATURES: {missing}")
            # Add them as zeros
            for f in missing:
                df[f] = 0
    
    try:
        probs = clf.predict_proba(df)[:, 1]
        print(f"{'y':>8} {'Angle':>8} {'xG':>8}")
        for i, y in enumerate(ys):
            print(f"{y:8.3f} {df.loc[i, 'angle_deg']:8.3f} {probs[i]:8.5f}")
            
    except Exception as e:
        print(f"Prediction error: {e}")

if __name__ == "__main__":
    probe_model()
