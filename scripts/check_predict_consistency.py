"""check_predict_consistency.py

Standalone script to verify that `analyze._predict_xgs` produces consistent results
when serving the XGBoost model, particularly for blocked shots.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from puck import fit_xgs, analyze, config as puck_config, fit_xgboost_nested

def check_consistency():
    print("--- Checking Prediction Consistency ---")
    
    # 1. Load Model
    model_path = Path('analysis/xgs/xg_model_nested_all.joblib')
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}. Run train_xgboost_model.py first.")
        # Fallback to default
        model_path = Path('analysis/xgs/xg_model_nested.joblib')
        
    print(f"Loading model from {model_path}...")
    try:
        clf = joblib.load(model_path)
        print(f"Model loaded: {type(clf).__name__}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 2. Load Sample Data (Raw)
    print("Loading sample data...")
    try:
        # Load just 2025-2026 for speed
        df_season = fit_xgs.load_data(str(Path('data/20252026/20252026_df.csv')))
    except Exception:
        # Fallback to general load if specific file missing
        df_season = fit_xgs.load_data()

    # Pick sample events including blocked shots
    # Ensure we get some blocked shots
    blocked = df_season[df_season['event'] == 'blocked-shot'].head(50)
    shots = df_season[df_season['event'] == 'shot-on-goal'].head(50)
    df_sample = pd.concat([blocked, shots], ignore_index=True)
    
    print(f"Sample size: {len(df_sample)}")

    # 3. Predict via analyze._predict_xgs
    print("Running analyze._predict_xgs...")
    
    # Force use of loaded model path
    # FIX SEED for stochastic imputation consistency
    np.random.seed(42)  
    df_pred, _, _ = analyze._predict_xgs(df_sample.copy(), model_path=str(model_path))
    
    # 4. Predict via Direct usage (fit_xgboost_nested)
    # This simulates "Training Time" logic
    print("Running training logic (test_predictions)...")
    # Preprocess
    df_train_logic = fit_xgboost_nested.preprocess_data(df_sample.copy())
    from puck import impute
    try:
        # FIX SEED SAME AS ABOVE
        np.random.seed(42)
        df_train_logic = impute.impute_blocked_shot_origins(df_train_logic, method='point_pull')
    except: pass
    
    # Predict directly
    probs_train = clf.predict_proba(df_train_logic)[:, 1]
    
    # 5. Compare
    # We need to align indices because preprocess_data might drop rows? 
    # analyze._predict_xgs maps back to original.
    
    print("\nComparing predictions...")
    
    # Align by index
    common_indices = df_train_logic.index.intersection(df_pred.index)
    
    # Extract
    pred_analyze = df_pred.loc[common_indices, 'xgs']
    pred_train = pd.Series(probs_train, index=df_train_logic.index).loc[common_indices]
    
    # Calculate Diff
    diff = (pred_analyze - pred_train).abs()
    
    if diff.max() > 1e-6:
        print("FAILURE: Significant difference found between analyze.py and training logic!")
        print(f"Max diff: {diff.max()}")
        print("bad rows:")
        print(diff[diff > 1e-6])
    else:
        print("SUCCESS: analyze.py matches training logic exactly.")
        
    # Check Blocked Shot Non-Zero
    blocked_preds = df_pred[df_pred['event'] == 'blocked-shot']['xgs']
    n_zero = (blocked_preds == 0).sum()
    print(f"\nBlocked Shot Verification:")
    print(f"  Count: {len(blocked_preds)}")
    print(f"  Mean xG: {blocked_preds.mean():.6f}")
    if n_zero == 0:
        print("  SUCCESS: No blocked shots have 0.0 xG (unless truly 0 probability).")
    else:
        print(f"  WARNING: {n_zero} blocked shots have 0.0 xG.")

if __name__ == "__main__":
    check_consistency()
