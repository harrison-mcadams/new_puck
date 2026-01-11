
import sys
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from puck import data_pipeline, rink

def verify():
    model_path = Path('analysis/xgs/xg_model_nested.joblib')
    model = joblib.load(model_path)
    
    # Grid in the O-Zone
    xs = np.linspace(25, 89, 65)
    ys = np.linspace(-42.5, 42.5, 86)
    xx, yy = np.meshgrid(xs, ys)
    
    grid_df = pd.DataFrame({
        'x': xx.ravel(),
        'y': yy.ravel(),
        'shooter_role': 'F',
        'event': 'shot-on-goal',
        'game_state': '5v5',
        'shot_type': 'Wrist Shot'
    })
    
    # Process
    grid_df = data_pipeline.preprocess_features(grid_df, is_training=False, apply_imputation=False)
    
    # Predict
    p_block = model.predict_proba_layer(grid_df, layer='block')
    zz = p_block.reshape(xx.shape)
    
    # Print max P in the "Slot" (x > 60, |y| < 15)
    mask_slot = (xx > 60) & (abs(yy) < 15)
    max_p_slot = np.max(zz[mask_slot])
    print(f"Max P(Blocked) in Slot (x > 60): {max_p_slot:.4f}")
    
    # Print P at a specific Slot point (75, 0)
    idx_75_0 = np.argmin((xx - 75)**2 + (yy - 0)**2)
    p_75_0 = p_block[idx_75_0]
    print(f"P(Blocked) at (75, 0): {p_75_0:.4f}")
    
    # Print P at a Point (30, 0)
    idx_30_0 = np.argmin((xx - 30)**2 + (yy - 0)**2)
    p_30_0 = p_block[idx_30_0]
    print(f"P(Blocked) at (30, 0): {p_30_0:.4f}")

if __name__ == "__main__":
    verify()
