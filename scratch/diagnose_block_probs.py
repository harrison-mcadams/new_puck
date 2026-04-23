import sys
import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from puck import fit_xgboost_alternate, config

def main():
    model_path = str(Path(config.ANALYSIS_DIR) / 'xgs' / 'xg_model_xgboost_alternate_modern_era.joblib')
    if not os.path.exists(model_path):
        print("Model not found.")
        return
        
    model = joblib.load(model_path)
    print("Block model features:", model.features_block)
    
    # Test cases: x=25 (Point) vs x=85 (Crease)
    test_points = [
        {'x': 25, 'y': 0, 'name': 'Low Slot / Point'},
        {'x': 50, 'y': 0, 'name': 'Middle Slot'},
        {'x': 85, 'y': 0, 'name': 'Crease'},
        {'x': 95, 'y': 0, 'name': 'Behind Net'}
    ]
    
    df_test = pd.DataFrame(test_points)
    
    # Add distance/angle
    from puck import rink
    dists, angles = [], []
    for i, row in df_test.iterrows():
        d, a = rink.calculate_distance_and_angle(row['x'], row['y'], 89.0, 0.0)
        dists.append(d)
        angles.append(a)
    df_test['distance'] = dists
    df_test['angle_deg'] = angles
    
    # Fill in categoricals manually using only needed ones
    from puck.fit_xgboost_alternate import CATEGORICAL_VOCABS
    for f in model.features_block:
        if f not in df_test.columns:
            if f in CATEGORICAL_VOCABS:
                df_test[f] = CATEGORICAL_VOCABS[f][0]
            else:
                df_test[f] = 0.0

    # Convert to categorical for XGBoost
    for col in CATEGORICAL_VOCABS.keys():
        if col in df_test.columns:
            df_test[col] = pd.Categorical(df_test[col], categories=CATEGORICAL_VOCABS[col])

    # EVALUATE
    # Use internal booster to be sure
    booster = model.model_block.get_booster()
    # Create DMatrix
    import xgboost as xgb
    dmatrix = xgb.DMatrix(df_test[model.features_block], enable_categorical=True)
    probs = booster.predict(dmatrix)
    
    print("\nBlock Model Predictions (Raw Booster):")
    for i, p in enumerate(probs):
        print(f"{test_points[i]['name']} (dist={df_test.at[i, 'distance']:.1f}): P(Block) = {p:.4f}")

if __name__ == "__main__":
    main()
