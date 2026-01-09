import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from puck import fit_xgboost_nested, features

def main():
    print("--- Block Model Leakage Investigation ---")
    
    # 1. Load Model
    model_path = 'analysis/xgs/xg_model_nested_all.joblib'
    try:
        clf = joblib.load(model_path)
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Inspect Block Model Features
    if not hasattr(clf, 'model_block'):
        print("Model does not have 'model_block' attribute.")
        return

    print("\nFeature Importances (Block Model):")
    model_block = clf.model_block
    # features are stored in clf.config_block.feature_cols
    feat_cols = clf.config_block.feature_cols
    
    if hasattr(model_block, 'feature_importances_'):
        imps = model_block.feature_importances_
        indices = np.argsort(imps)[::-1]
        
        for i in range(min(20, len(indices))):
            idx = indices[i]
            print(f"  {i+1}. {feat_cols[idx]}: {imps[idx]:.4f}")
            
        top_feature = feat_cols[indices[0]]
    else:
        print("  Model does not provide feature_importances_")

    # 3. Load Data Sample to check distributions
    print("\nLoading sample data for distribution check...")
    # Load the test predictions CSV if available as it has processed features?
    # Or load raw data and process it similarly.
    # Let's try loading a raw file to see "natural" distributions vs imputed.
    
    try:
        # Load a recent season file
        df = pd.read_csv('data/20232024/20232024_df.csv')
        print(f"Loaded {len(df)} rows from 20232024.")
        
        # We need to apply the same imputation to see what the model sees
        from puck import impute, correction
        
        df = correction.fix_blocked_shot_attribution(df)
        
        # Apply imputation
        # Check standard cols
        if 'x_adj' in df.columns:
            x_col, y_col = 'x_adj', 'y_adj'
        else:
            x_col, y_col = 'x', 'y'
            
        print(f"Imputing with {x_col}, {y_col}...")
        df_imputed = impute.impute_blocked_shot_origins(df, method='empirical_model', x_col=x_col, y_col=y_col)
        
        # Label
        df_imputed['is_blocked'] = (df_imputed['event'] == 'blocked-shot').astype(int)
        
        # Filter for valid shot attempts only
        valid_shots = ['shot-on-goal', 'goal', 'missed-shot', 'blocked-shot']
        df_imputed = df_imputed[df_imputed['event'].isin(valid_shots)].copy()
        
        # 4. Compare Distributions of Top Features
        # Specifically Distance and Angle
        mask_blocked = df_imputed['is_blocked'] == 1
        mask_unblocked = df_imputed['is_blocked'] == 0
        
        stats_path = 'analysis/leakage_stats.txt'
        with open(stats_path, 'w') as f:
            f.write("Distribution Summary (Mean / Std):\n")
            for col in ['distance', 'angle_deg', 'time_elapsed_in_period_s', 'score_diff']:
                if col in df_imputed.columns:
                    m_b = df_imputed.loc[mask_blocked, col].mean()
                    s_b = df_imputed.loc[mask_blocked, col].std()
                    m_u = df_imputed.loc[mask_unblocked, col].mean()
                    s_u = df_imputed.loc[mask_unblocked, col].std()
                    f.write(f"  {col}:\n")
                    f.write(f"    Blocked:   {m_b:.2f} +/- {s_b:.2f}\n")
                    f.write(f"    Unblocked: {m_u:.2f} +/- {s_u:.2f}\n")
        
        print(f"Saved distribution stats to {stats_path}")

        # Check for Overlap / Distinctness in Distance
        # If blocked distance is always approx X, that's a leak/proxy.
        
        # Histogram for Distance
        plt.figure(figsize=(10, 6))
        plt.hist(df_imputed.loc[mask_unblocked, 'distance'].dropna(), bins=50, alpha=0.5, label='Unblocked', density=True)
        plt.hist(df_imputed.loc[mask_blocked, 'distance'].dropna(), bins=50, alpha=0.5, label='Blocked', density=True)
        plt.title("Distance Distribution: Blocked vs Unblocked")
        plt.xlabel("Distance")
        plt.legend()
        plt.savefig('analysis/nested_xgs/debug_distance_dist.png')
        print("Saved distance distribution plot to analysis/nested_xgs/debug_distance_dist.png")

        # Histogram for Angle
        plt.figure(figsize=(10, 6))
        plt.hist(df_imputed.loc[mask_unblocked, 'angle_deg'].dropna(), bins=50, alpha=0.5, label='Unblocked', density=True)
        plt.hist(df_imputed.loc[mask_blocked, 'angle_deg'].dropna(), bins=50, alpha=0.5, label='Blocked', density=True)
        plt.title("Angle Distribution: Blocked vs Unblocked")
        plt.xlabel("Angle")
        plt.legend()
        plt.savefig('analysis/nested_xgs/debug_angle_dist.png')
        print("Saved angle distribution plot to analysis/nested_xgs/debug_angle_dist.png")

    except Exception as e:
        print(f"Error during data analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
