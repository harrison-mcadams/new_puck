import sys
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import fit_xgboost_nested, fit_xgs

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
            
            print(f"{'Feature':<20} | {'Importance':<10}")
            print("-" * 35)
            for i in indices:
                print(f"{features[i]:<20} | {importances[i]:.4f}")
        else:
            print("Model does not have feature_importances_ attribute.")
    else:
        print("Model does not have model_block or config_block attributes.")

    # 2. Check Correlations in Data
    print("\n--- Correlation Analysis ---")
    print("Loading sample data...")
    df = fit_xgs.load_data().sample(min(50000, 30000), random_state=42) # Valid subsample
    
    print("Enriching with bios...")
    df = fit_xgs.enrich_data_with_bios(df)
    
    # Create target
    df['is_blocked'] = (df['event'] == 'blocked-shot').astype(int)
    
    # Encode shooter_role
    # F -> 0, D -> 1
    df['role_D'] = (df['shooter_role'] == 'D').astype(int)
    
    # Calculate simple correlation
    corr = df['role_D'].corr(df['is_blocked'])
    print(f"Correlation (Shooter Role=D vs Blocked): {corr:.4f}")
    
    # Check block rate by role
    rate_D = df[df['shooter_role'] == 'D']['is_blocked'].mean()
    rate_F = df[df['shooter_role'] == 'F']['is_blocked'].mean()
    
    print(f"Block Rate (Defenders): {rate_D:.4f}")
    print(f"Block Rate (Forwards):  {rate_F:.4f}")

    # Check validation of features used
    print(f"\nFeatures used in Block Model: {clf.config_block.feature_cols}")

if __name__ == "__main__":
    investigate()
