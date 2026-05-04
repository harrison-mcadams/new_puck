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
    
    # Let's generate a synthetic dataset of straight-on shots at various distances
    distances = np.linspace(5, 60, 12)
    
    data = []
    for d in distances:
        # Assuming straight on from net: x = 89 - d, y = 0
        x = 89 - d
        row = {
            'x': x,
            'y': 0,
            'distance': d,
            'angle_deg': 0,
            'shot_type': 'wrist',
            'shooter_role': 'F',
            'shoots_catches': 'L',
            'game_state': '5v5',
            'relative_game_state': '5v5',
            'is_rush': 0,
            'is_rebound': 0,
            'is_home': 1,
            'score_diff': 0,
            'period_number': 1,
            'speed_from_last_event': 10.0,
            'last_event_type': 'faceoff',
            'dist_from_last_event': 20.0,
            'last_event_time_diff': 5.0
        }
        data.append(row)
        
    df = pd.DataFrame(data)
    
    # Prepare inference df handles splines and encoding
    df_inf = model._prepare_inference_df(df)
    
    # Predict finish probability
    p_finish = model._predict_marginalized(model.model_finish, df_inf, model.features_fin)
    
    print("Distance | Finish Prob")
    print("----------------------")
    for d, p in zip(distances, p_finish):
        print(f"{d:8.1f} | {p:10.4f}")

if __name__ == '__main__':
    main()
