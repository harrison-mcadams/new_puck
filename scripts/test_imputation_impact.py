"""test_imputation_impact.py"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from puck import fit_xgs, data_pipeline, fit_glm_nested, features as feature_util

def main():
    print("Loading Data (10k sample)...")
    df = fit_xgs.load_data()
    # Ensure we get blocked shots
    blocks = df[df['event'] == 'blocked-shot'].sample(min(5000, len(df)), random_state=42)
    others = df[df['event'] != 'blocked-shot'].sample(min(5000, len(df)), random_state=42)
    df_sample = pd.concat([blocks, others])
    
    print(f"Sample size: {len(df_sample)}")

    # Load Model
    model_path = Path('analysis/xgs/xg_model_nested.joblib')
    print(f"Loading model from {model_path}...")
    clf = joblib.load(model_path)
    
    # Scenario A: With Imputation (Pipeline standard)
    print("\n--- Scenario A: With Empirical Imputation ---")
    df_A = data_pipeline.preprocess_features(
        df_sample.copy(), 
        is_training=False, 
        apply_imputation=True,
        apply_arena_adjustments=True
    )
    probs_A = clf.predict_proba(df_A)[:, 1]
    
    print(f"Mean xG: {probs_A.mean():.4f}")
    print(f"Max xG: {probs_A.max():.4f}")
    print(f"Count > 0.3: {(probs_A > 0.3).sum()}")
    print(f"Count > 0.5: {(probs_A > 0.5).sum()}")
    
    # Scenario B: Without Imputation (Raw Block Coords)
    print("\n--- Scenario B: Without Imputation (Raw Block Coords) ---")
    # We disable imputation. 
    # Note: data_pipeline logic must be checked. 
    # If apply_imputation=False, 'x_adj' comes from 'x'.
    # For blocked shots, 'x' is the block location?
    # We must ensure 'correction.fix_blocked_shot_attribution' doesn't mess it up if we want Pure Raw.
    # But let's trust pipeline flag.
    df_B = data_pipeline.preprocess_features(
        df_sample.copy(), 
        is_training=False, 
        apply_imputation=False, # DISABLED
        apply_arena_adjustments=True
    )
    # Note: If imputation disabled, blocked shots might use Block Location (near net?).
    # Let's verify distance.
    print(f"Mean Distance (Scenario B): {df_B[df_B['event']=='blocked-shot']['distance'].mean():.1f}")
    print(f"Mean Distance (Scenario A): {df_A[df_A['event']=='blocked-shot']['distance'].mean():.1f}")
    
    probs_B = clf.predict_proba(df_B)[:, 1]
    
    print(f"Mean xG: {probs_B.mean():.4f}")
    print(f"Max xG: {probs_B.max():.4f}")
    print(f"Count > 0.3: {(probs_B > 0.3).sum()}")
    print(f"Count > 0.5: {(probs_B > 0.5).sum()}")

if __name__ == "__main__":
    main()
