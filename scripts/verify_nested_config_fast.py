
import sys
import os
import pandas as pd
import numpy as np
import joblib
from unittest.mock import patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import fit_nested_xgs, analyze

def create_dummy_data():
    # Create enough rows to split and train
    n = 100
    df = pd.DataFrame({
        'event': np.random.choice(['shot-on-goal', 'blocked-shot', 'missed-shot', 'goal'], n),
        'distance': np.random.uniform(10, 60, n),
        'angle_deg': np.random.uniform(-45, 45, n),
        'game_state': ['5v5'] * n,
        'is_net_empty': [0] * n,
        'shot_type': np.random.choice(['Wrist Shot', 'Slap Shot', 'Snap Shot'], n),
        'x': np.random.uniform(-100, 100, n),
        'y': np.random.uniform(-42, 42, n)
    })
    return df

def verify_pipeline():
    print("--- Verifying fit_nested_xgs.py with MOCKED DATA ---")
    
    # Patch load_data to return dummy data
    with patch('puck.fit_nested_xgs.load_data', side_effect=create_dummy_data) as mock_load:
        try:
            # Run main to train and save (fast)
            fit_nested_xgs.main()
            print("Training finished (mocked).")
            
            # Check if file exists
            model_path = 'analysis/xgs/xg_model_nested.joblib'
            if os.path.exists(model_path):
                print(f"SUCCESS: {model_path} created.")
            else:
                print(f"FAILURE: {model_path} not found.")
                return False
                
            # Try loading it via analyze
            print("\n--- Verifying analyze._predict_xgs ---")
            df_test = create_dummy_data()
            df_pred, clf, meta = analyze._predict_xgs(df_test)
            
            if 'xgs' in df_pred.columns:
                print("SUCCESS: 'xgs' column added.")
                print(df_pred[['event', 'xgs']].head())
            else:
                print("FAILURE: 'xgs' column missing.")
                return False
                
            if 'NestedXGClassifier' in str(type(clf)):
                 print("SUCCESS: Loaded NestedXGClassifier by default.")
            else:
                 print(f"WARNING: Loaded {type(clf)}, expected NestedXGClassifier.")
                 
            return True
        except Exception as e:
            print(f"FAILURE in pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    if verify_pipeline():
        print("\nALL CHECKS PASSED")
        sys.exit(0)
    else:
        print("\nCHECKS FAILED")
        sys.exit(1)
