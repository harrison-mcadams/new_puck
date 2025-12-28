"""debug_finish_model.py

Script to check:
1. Shot Type distribution in training data (Finish Layer).
2. Effect of changing shot_type from NaN to 'Wrist Shot' on blocked shot finish probability.
"""

import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))
from puck import fit_xgboost_nested

def debug():
    print("--- Debugging Finish Model ---")
    
    # 1. Load Model
    model_path = Path('analysis/xgs/xg_model_nested_all.joblib')
    if not model_path.exists():
        print("Model not found.")
        return
        
    clf = joblib.load(model_path)
    finish_model = clf.model_finish
    
    if finish_model is None:
        print("Finish model is None.")
        return
    
    print("Model loaded.")
    
    # 2. Check Training Data Shot Types (Infer from Model booster or load data?)
    # Loading data takes time. Let's just test the prediction behavior first.
    
    # 3. Create Synthetic Data (Simulating a high-prob blocked shot)
    # Based on Row 290055: dist=29, angle=68, type=NaN
    
    data = {
        'distance': [29.0, 29.0, 29.0],
        'angle_deg': [68.0, 68.0, 68.0],
        'game_state': ['5v5', '5v5', '5v5'],
        'shoots_catches': ['L', 'L', 'L'],
        'shot_type': [np.nan, 'Wrist Shot', 'Slap Shot'] 
    }
    
    df_test = pd.DataFrame(data)
    
    # Preprocess (Categorical Casting)
    for c in ['game_state', 'shoots_catches', 'shot_type']:
        df_test[c] = df_test[c].astype('category')
        
    # Predict with Finish Model
    feature_names = finish_model.get_booster().feature_names
    # df_test = df_test[feature_names] # Wrapper handles column selection if we pass full/superset? 
    # Actually wrapper _prepare_data expects 'features' list. 
    # But predict_proba_layer selects cols.
    # So we don't strictly need to subset if predict_proba_layer does it.
    # Let's ensure cols exist.
    pass
    
    print("\n--- Prediction Test ---")
    print(df_test)
    
    try:
        # Predict using wrapper method (which calls _prepare_data and applies the fix)
        preds = clf.predict_proba_layer(df_test, 'finish')
        
        df_test['prob_finish'] = preds
        print("\nResults:")
        print(df_test[['shot_type', 'prob_finish']])
        
    except Exception as e:
        print(f"Prediction failed: {e}")
        # Maybe feature mismatch?
        print(f"Model features: {finish_model.get_booster().feature_names}")

if __name__ == "__main__":
    debug()
