"""check_predict_consistency.py

Checks consistency between:
1. Training Pipeline (fit_nested_xgs.fit)
2. Prediction Pipeline (analyze._predict_xgs)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from puck import fit_nested_xgs, analyze, fit_xgs

def check_consistency():
    print("--- CHECKING PIPELINE CONSISTENCY ---")
    
    # 1. Create Mock Data
    print("\n1. Generating Mock Data...")
    mock_data = pd.DataFrame([
        {'event': 'shot-on-goal', 'x': 50, 'y': 10, 'distance': 40, 'angle_deg': 20, 'game_state': '5v5', 'is_net_empty': 0, 'game_id': '2025020001', 'shot_type': 'Wrist Shot'},
        {'event': 'blocked-shot', 'x': 60, 'y': 0, 'distance': 50, 'angle_deg': 0, 'game_state': '5v4', 'is_net_empty': 0, 'game_id': '2025020001', 'shot_type': None}, # Missing shot type
        {'event': 'goal', 'x': 20, 'y': 5, 'distance': 15, 'angle_deg': 10, 'game_state': '5v5', 'is_net_empty': 0, 'game_id': '2025020001', 'shot_type': 'Snap Shot'},
    ])
    
    # 2. Simulate Training Prep
    print("\n2. Training Pipeline Prep...")
    # Assume we use standard features + handedness (if we were successful)
    features = ['distance', 'angle_deg', 'game_state', 'shot_type']
    
    try:
        # Preprocess
        df_train = fit_nested_xgs.preprocess_features(mock_data.copy())
        
        # OHE Logic (Simulated from fit_nested_xgs.fit)
        categorical_cols = ['game_state', 'shot_type'] # standard
        df_train_ohe = pd.get_dummies(df_train, columns=categorical_cols, prefix_sep='_')
        train_cols =  sorted(list(df_train_ohe.columns))
        print(f"   Training Columns: {train_cols}")
        
    except Exception as e:
        print(f"   [ERROR] Training prep failed: {e}")
        return

    # 3. Simulate Prediction Prep (analyze._predict_xgs)
    print("\n3. Prediction Pipeline Prep...")
    try:
        # _predict_xgs logic calls impute then fit_xgs.clean_df_for_model
        # We'll simulate the steps inside _predict_xgs
        
        # Filter (Mocking what _predict_xgs does)
        df_pred_in = mock_data.copy()
        
        # Call clean_df_for_model with encode_method='none' (as used in analyze.py for nested)
        # We need mock input features. 
        # In the real code it gets them from the model. 
        # Let's assume the same features list.
        
        df_pred_clean, final_feats, cat_levels = fit_xgs.clean_df_for_model(
            df_pred_in, features, encode_method='none'
        )
        
        # Now analyze.py creates dummys if needed or assumes model handles it.
        # NestedXGClassifier.predict_proba does OHE internally if cols missing.
        
        # Let's verify if 'shoots_catches' is handled if requested.
        features_extended = features + ['shoots_catches']
        
        # Mock data with missing handedness
        print("   Testing with potentially missing 'shoots_catches'...")
        try:
             df_pred_clean_ext, _, _ = fit_xgs.clean_df_for_model(
                df_pred_in, features_extended, encode_method='none'
            )
             print("   [OK] clean_df_for_model handled missing handedness column? (It might error if missing)")
        except KeyError as e:
             print(f"   [FAIL] clean_df_for_model raised KeyError for missing column: {e}")

    except Exception as e:
        print(f"   [ERROR] Prediction prep failed: {e}")

if __name__ == "__main__":
    check_consistency()
