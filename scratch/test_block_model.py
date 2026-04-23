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
    print("Loaded model features:", model.features)
    
    # Test cases
    test_points = [
        {'x': 25, 'y': 0, 'name': 'Low Slot / Point'},
        {'x': 50, 'y': 0, 'name': 'Middle Slot'},
        {'x': 85, 'y': 0, 'name': 'Crease'},
        {'x': 95, 'y': 0, 'name': 'Behind Net'},
        {'x': 95, 'y': 25, 'name': 'Behind Net Corner'}
    ]
    
    df_test = pd.DataFrame(test_points)
    # Add defaults for other features
    from puck.fit_xgboost_alternate import CATEGORICAL_VOCABS
    
    for f in model.features:
        if f not in df_test.columns:
            if f in CATEGORICAL_VOCABS:
                df_test[f] = CATEGORICAL_VOCABS[f][0]
            else:
                df_test[f] = 0.0
            
    df_test['shot_type'] = 'wrist'
    df_test['relative_game_state'] = '5v5'
    df_test['last_event_type'] = 'giveaway'

    # Convert to category
    for col in CATEGORICAL_VOCABS.keys():
        if col in df_test.columns:
            df_test[col] = df_test[col].astype('category')
    
    # Calculate distance/angle
    from puck import rink
    for i, row in df_test.iterrows():
        d, a = rink.calculate_distance_and_angle(row['x'], row['y'], 89.0, 0.0)
        df_test.at[i, 'distance'] = d
        df_test.at[i, 'angle_deg'] = a

    # Predict
    # The models are model.model_block, model.model_acc, model.model_finish
    def sigmoid(x): return 1 / (1 + np.exp(-x))
    
    # We need to use model._prepare_dataframe if it exists to handle categoricals
    # But model.model_block.predict_proba is easier
    
    prob_block = model.model_block.predict_proba(df_test[model.features])[:, 1]
    
    print("\nModel Evaluation (Block Probability):")
    for i, p in enumerate(prob_block):
        print(f"{test_points[i]['name']} (x={test_points[i]['x']}, y={test_points[i]['y']}, dist={df_test.at[i, 'distance']:.1f}): P(Block) = {p:.4f}")

if __name__ == "__main__":
    main()
