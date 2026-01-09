import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from puck import analyze, fit_xgs, correction, impute, fit_xgboost_nested, config as p_conf

def verify_consistency():
    print("--- Verifying Pipeline Consistency ---")
    
    # 1. Load a small sample of raw data
    # We want a mix of events to test all paths
    print("Loading sample data...")
    df_raw = fit_xgs.load_data()
    
    # Get a deterministic sample containing blocked shots and other events
    mask_blocked = df_raw['event'] == 'blocked-shot'
    mask_goal = df_raw['event'] == 'goal'
    mask_shot = df_raw['event'] == 'shot-on-goal'
    
    # Take 5 of each if available, else whatever we have
    df_sample = pd.concat([
        df_raw[mask_blocked].head(5),
        df_raw[mask_goal].head(5),
        df_raw[mask_shot].head(5)
    ], ignore_index=True)
    
    print(f"Sample size: {len(df_sample)}")
    
    # ---------------------------------------------------------
    # Path A: Inference Pipeline (The "Production" Path)
    # ---------------------------------------------------------
    print("\nRunning Path A: Inference (puck.analyze._predict_xgs)...")
    # This function handles correction, imputation, and prediction internally
    df_inf, clf_inf, _ = analyze._predict_xgs(df_sample, behavior='overwrite')
    
    # ---------------------------------------------------------
    # Path B: Training Pipeline (The "Training" Path)
    # ---------------------------------------------------------
    print("\nRunning Path B: Training Pipeline Steps...")
    df_train = df_sample.copy()
    
    # B1. Fix Attribution
    if 'event' in df_train.columns and 'blocked-shot' in df_train['event'].unique():
        df_train = correction.fix_blocked_shot_attribution(df_train)
        
    # B2. Impute (Empirical + Adj) - Explicitly replicating training logic
    suffix = getattr(p_conf, 'COORDINATE_SUFFIX', '_adj')
    cx, cy = f"x{suffix}", f"y{suffix}"
    use_x, use_y = 'x', 'y'
    if cx in df_train.columns and cy in df_train.columns:
        print(f"  Training Logic: Using Adjusted Coordinates: {cx}, {cy}")
        use_x, use_y = cx, cy
    
    df_train = impute.impute_blocked_shot_origins(df_train, method='empirical_model', x_col=use_x, y_col=use_y)
    
    # B3. Preprocess (Categoricals, etc.)
    # We need to simulate the model's internal usage.
    # The loaded model in analyze is the same one we trained.
    # But analyze calls clf.predict_proba(df) directly after imputation.
    # The XGBNestedXGClassifier.predict_proba calls self._prepare_df()
    
    # So actually, if analyze just does correction + imputation, then passes to predict_proba,
    # and we do correction + imputation here, the inputs to predict_proba should be identical.
    
    # Let's compare the DataFrames just before prediction would happen.
    # analyze returns the DF *after* prediction, so it has xgs.
    # But it also keeps the imputed columns? 
    # Let's check if analyze modifies in place or returns new. It copies.
    
    # We mainly want to verify that the coordinate features used for the model are identical.
    # The model uses 'distance' and 'angle_deg'.
    
    cols_to_check = ['distance', 'angle_deg']
    
    print("\n--- Comparison ---")
    
    # 1. Check Features
    for col in cols_to_check:
        vals_inf = df_inf[col].fillna(0).values
        vals_train = df_train[col].fillna(0).values
        
        is_close = np.allclose(vals_inf, vals_train, atol=1e-5)
        if is_close:
            print(f"[PASS] Feature '{col}' matches exactly.")
        else:
            print(f"[FAIL] Feature '{col}' mismatch!")
            print("Inference:", vals_inf)
            print("Training: ", vals_train)
            print("Diff:     ", vals_inf - vals_train)

    # 2. Check Predictions
    # We can't easily reproduce the exact prediction call without loading the model again,
    # so we'll rely on the feature check. If features are same, model is deterministic.
    # But let's check if the Imputed Coordinates match.
    
    if 'imputed_x' in df_inf.columns:
        # Check specific rows that were blocked shots
        blocked_idxs = df_sample[df_sample['event'] == 'blocked-shot'].index
        
        # We need to find the corresponding rows in the processed outputs.
        # Since we concatenated, indices might be reset. 
        # Actually validation is easier if we just rely on order since we processed the same df.
        
        vals_inf_imp_x = df_inf.loc[blocked_idxs, 'imputed_x'].fillna(0).values
        vals_train_imp_x = df_train.loc[blocked_idxs, 'imputed_x'].fillna(0).values
        
        if np.allclose(vals_inf_imp_x, vals_train_imp_x, atol=1e-5, equal_nan=True):
             print("[PASS] Imputed X coordinates match for blocked shots.")
        else:
             print("[FAIL] Imputed X mismatch!")
             print("Inf:", vals_inf_imp_x)
             print("Trn:", vals_train_imp_x)
             
    print("\nVerification Complete.")

if __name__ == "__main__":
    verify_consistency()
