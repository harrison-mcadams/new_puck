
import sys
import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Fix import paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from puck import config, fit_nested_xgs, fit_xgboost_nested

# --- HACK FOR JOBLIB LOADING ---
if not hasattr(sys.modules['__main__'], 'NestedXGClassifier'):
    setattr(sys.modules['__main__'], 'NestedXGClassifier', fit_nested_xgs.NestedXGClassifier)
if not hasattr(sys.modules['__main__'], 'LayerConfig'):
    setattr(sys.modules['__main__'], 'LayerConfig', fit_nested_xgs.LayerConfig)
if hasattr(fit_xgboost_nested, 'XGBNestedXGClassifier'):
    if not hasattr(sys.modules['__main__'], 'XGBNestedXGClassifier'):
        setattr(sys.modules['__main__'], 'XGBNestedXGClassifier', fit_xgboost_nested.XGBNestedXGClassifier)

def main():
    print("Starting Calibration Investigation...")
    
    # 1. Load Model
    model_path = Path(config.ANALYSIS_DIR) / 'xgs' / 'xg_model_nested.joblib'
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Load Data
    print("Loading data...")
    df = fit_nested_xgs.load_data()
    df = fit_nested_xgs.preprocess_features(df)
    
    # Impute
    from puck import impute
    df = impute.impute_blocked_shot_origins(df, method='point_pull')
    
    # Split (Same validation set)
    _, df_test = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Validation Set: {len(df_test)} rows")
    
    # Ensure columns exist
    for c in model.final_features:
        if c not in df_test.columns:
            df_test[c] = 0
            
    # 3. Predict Overall xG
    print("Predicting probabilities...")
    probs = model.predict_proba(df_test)[:, 1]
    df_test['xG'] = probs
    df_test['is_goal_final'] = (df_test['event'] == 'goal').astype(int)
    
    # 4. Analyze bins across the spectrum to see if underestimation is systemic
    bins = [
        (0.0, 0.05),
        (0.05, 0.15),
        (0.15, 0.25),
        (0.25, 0.50),
        (0.50, 1.00)
    ]
    
    print(f"\n{'Bin Range':<15} | {'Count':<7} | {'Pred':<8} | {'Actual':<8} | {'Diff':<8} | {'Status':<15}")
    print("-" * 85)
    
    for bin_low, bin_high in bins:
        mask_bin = (df_test['xG'] >= bin_low) & (df_test['xG'] < bin_high)
        df_bin = df_test[mask_bin].copy()
        
        n = len(df_bin)
        if n == 0:
            print(f"[{bin_low:.2f}, {bin_high:.2f}) | {0:<7} | {'-':<8} | {'-':<8} | {'-':<8} | -")
            continue
            
        pred = df_bin['xG'].mean()
        actual = df_bin['is_goal_final'].mean()
        diff = actual - pred
        
        status = "Underestimated" if diff > 0.01 else ("Overestimated" if diff < -0.01 else "Calibrated")
        # Add emphasis for big misses
        if abs(diff) > 0.05: status += " (!)"
        
        print(f"[{bin_low:.2f}, {bin_high:.2f}) | {n:<7} | {pred:.4f}   | {actual:.4f}   | {diff:+.4f}   | {status}")
        
    print("-" * 85)

if __name__ == "__main__":
    main()
