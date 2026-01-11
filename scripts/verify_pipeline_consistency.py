import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from puck import analyze, fit_xgs, correction, impute, fit_xgboost_nested, config as p_conf, data_pipeline

def verify_consistency():
    print("--- Verifying Pipeline Consistency ---")
    
    # 1. Load a small sample of raw data
    # We want a mix of events to test all paths
    print("Loading sample data (20232024)...")
    # Optimize: Load single season directly
    try:
        data_path = Path(__file__).resolve().parent.parent / "data" / "20232024" / "20232024_df.csv"
        df_raw = pd.read_csv(data_path)
    except Exception as e:
        print(f"Failed to load season, falling back to all data: {e}")
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
    
    # B1. Unified Pipeline (Simulating Training/Production Logic)
    # The "Training" path in verify_pipeline_consistency usually checks if
    # manually applying steps matches the "Inference" function.
    # Now both use data_pipeline.
    
    print("  Applying data_pipeline.preprocess_features (is_training=False)...")
    df_train = data_pipeline.preprocess_features(
        df_train,
        is_training=False, # Match inference for consistency check
        apply_arena_adjustments=True,
        apply_imputation=True,
        apply_dithering=False
    )
    
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
    
    # 1. Check Features (Restrict to Shot Attempts)
    # analyze.py only runs pipeline on shots. Training pipeline (in verification) ran on everything.
    # So we care only about shots matching.
    shot_types = ['shot-on-goal', 'goal', 'missed-shot', 'blocked-shot']
    # Filter both DFs to just shots for comparison (using df_inf's classification if needed, or just event column)
    
    # We assume 'event' column is consistent.
    mask_inf = df_inf['event'].isin(shot_types)
    # df_train might have 'event' too.
    
    print(f"  Comparing features for {mask_inf.sum()} shot events...")
    
    for col in cols_to_check:
        # subset to mask
        vals_inf = df_inf.loc[mask_inf, col].fillna(0).values
        vals_train = df_train.loc[mask_inf, col].fillna(0).values
        
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
             
    # 3. Sanity Check: Distances
    # Ensure we don't have cross-rink distances (e.g. > 100ft) for offensive zone events.
    # Blocked shots should be relatively close to the net or at least in the zone (< 89+25 ~ 114? No, zone is 64ft long).
    
    if 'distance' in df_inf.columns:
         blocked_dists = df_inf.loc[df_inf['event'] == 'blocked-shot', 'distance']
         if not blocked_dists.empty:
             max_dist = blocked_dists.max()
             mean_dist = blocked_dists.mean()
             print(f"\nBlocked Shot Distance Check: Max={max_dist:.1f} ft, Mean={mean_dist:.1f} ft")
             if max_dist > 100:
                 print("[WARN] Found blocked shots with distance > 100 ft! Orientation might be wrong.")
             else:
                 print("[PASS] Blocked shot distances look reasonable (< 100 ft).")
                 
    # 4. Check Coordinate Swap (in Pipeline Output)
    # We check df_train because it comes directly from preprocess_features.
    # df_inf comes from analyze.py which might preserve raw x/y.
    
    if 'x_adj' in df_train.columns and 'x' in df_train.columns:
        # Check if x == x_adj (ignoring nans)
        mismatches = 0
        try:
             # Fill na with -999 for comparison
             v1 = df_train['x'].fillna(-999)
             v2 = df_train['x_adj'].fillna(-999)
             mismatches = (v1 != v2).sum()
        except:
             mismatches = -1
             
        if mismatches == 0:
            print("[PASS] Final 'x' column matches 'x_adj' in pipeline output.")
        else:
            print(f"[WARN] Final 'x' does not match 'x_adj' in pipeline output! Mismatches: {mismatches}")

    print("\nVerification Complete.")

if __name__ == "__main__":
    verify_consistency()
