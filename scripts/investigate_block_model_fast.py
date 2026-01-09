import sys
import os
import joblib
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import fit_xgboost_nested

def investigate():
    model_path = 'analysis/xgs/xg_model_nested_all.joblib'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    clf = joblib.load(model_path)
    
    # 1. Check Feature Importances for Block Model
    print("\n--- Block Model Feature Importances ---")
    if hasattr(clf, 'model_block') and hasattr(clf, 'config_block'):
        model = clf.model_block
        features = clf.config_block.feature_cols
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            print(f"{'Feature':<25} | {'Importance':<10}")
            print("-" * 40)
            for i in indices:
                print(f"{features[i]:<25} | {importances[i]:.4f}")
        else:
            print("Model does not have feature_importances_ attribute.")
    else:
        print("Model does not have model_block or config_block attributes.")

if __name__ == "__main__":
    investigate()
