import pandas as pd
import numpy as np
import joblib
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import config as puck_config, fit_xgboost_tensor

def main():
    model_path = str(Path(puck_config.ANALYSIS_DIR) / 'xgs' / 'xg_model_xgboost_tensor.joblib')
    if not os.path.exists(model_path):
        print("Model not found")
        return

    model = joblib.load(model_path)
    
    grid_x = np.linspace(0, 100, 50)
    grid_y = np.linspace(-42.5, 42.5, 43)
    xx, yy = np.meshgrid(grid_x, grid_y)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    
    df = pd.DataFrame(points, columns=['x', 'y'])
    df['distance'] = np.sqrt((np.clip(df['x'], 0, 100) - 89)**2 + np.clip(df['y'], -42.5, 42.5)**2)
    angle_rad = np.arctan2(np.clip(df['x'], 0, 100) - 89, -np.clip(df['y'], -42.5, 42.5))
    df['angle_deg'] = ((-angle_rad * 180 / np.pi) % 360 + 360) % 360
    
    # Fill in defaults
    defaults = {
        'shot_type': 'wrist', 'shooter_role': 'F', 'shoots_catches': 'L',
        'game_state': '5v5', 'relative_game_state': '5v5',
        'is_rush': 0, 'is_rebound': 0, 'is_home': 1, 'score_diff': 0,
        'period_number': 2, 'speed_from_last_event': 7.5, 'last_event_type': 'giveaway',
        'dist_from_last_event': 15.0, 'last_event_time_diff': 2.0
    }
    for k, v in defaults.items():
        df[k] = v
        
    df_inf = model._prepare_inference_df(df)
    p_finish = model._predict_marginalized(model.model_finish, df_inf, model.features_fin)
    
    # Calculate some stats
    print(f"Mean Finish Prob: {p_finish.mean():.4f}")
    print(f"Max Finish Prob: {p_finish.max():.4f}")
    print(f"Min Finish Prob: {p_finish.min():.4f}")
    print(f"Percent of ice with finish > 0.2: {(p_finish > 0.2).mean() * 100:.2f}%")
    print(f"Percent of ice with finish > 0.5: {(p_finish > 0.5).mean() * 100:.2f}%")
    
    # Look at specific zones
    crease_mask = (df['x'] > 85) & (df['y'].abs() < 5)
    slot_mask = (df['x'] > 70) & (df['x'] <= 85) & (df['y'].abs() < 15)
    point_mask = (df['x'] > 25) & (df['x'] <= 70) & (df['y'].abs() > 15)
    
    print(f"Crease Avg: {p_finish[crease_mask].mean():.4f}")
    print(f"Slot Avg: {p_finish[slot_mask].mean():.4f}")
    print(f"Point Avg: {p_finish[point_mask].mean():.4f}")
    print(f"Outside Avg: {p_finish[~(crease_mask | slot_mask | point_mask)].mean():.4f}")

if __name__ == '__main__':
    main()
