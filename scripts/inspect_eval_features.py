import sys
import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from puck import analyze, data_pipeline

def inspect_evaluation_features():
    model_path = 'analysis/xgs/xg_model_xgboost_alternate_modern_era.joblib'
    if not os.path.exists(model_path):
        print("Model not found.")
        return
        
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    
    # Load some real data
    csv_path = analyze.locate_season_csv('20232024')
    df = pd.read_csv(csv_path).head(100)
    df_p = data_pipeline.preprocess_features(df, apply_filtering=True)
    
    # Process through the model's own inference pipeline
    df_inf = model._prepare_inference_df(df_p)
    
    # Focus on Block Model features
    feats = model.features_block
    print(f"\nInspecting {len(feats)} features for the Block Submodel:")
    
    # Take a sample row that is a blocked shot
    block_rows = df_p[df_p['event'] == 'blocked-shot']
    if len(block_rows) > 0:
        sample_idx = block_rows.index[0]
        print(f"\n--- Feature Values for a REAL Blocked Shot (Index {sample_idx}) ---")
        row = df_inf.loc[sample_idx]
        for f in feats:
            val = row[f]
            # Check if it's a category
            if isinstance(df_inf[f].dtype, pd.CategoricalDtype):
                cat_val = val
                code = df_inf[f].cat.codes.loc[sample_idx]
                print(f"{f:25}: {cat_val} (Code: {code})")
            else:
                print(f"{f:25}: {val}")
    else:
        print("\nNo blocked shots found in the first 100 rows of 20232024.")
        # Just use the first row
        row = df_inf.iloc[0]
        for f in feats:
             print(f"{f:25}: {row[f]}")

    # Check for NaNs globally in the evaluation set
    print("\n--- Global Feature Health (First 100 rows) ---")
    nan_counts = df_inf[feats].isna().sum()
    if nan_counts.sum() > 0:
        print("WARNING: Found NaNs in evaluation features!")
        print(nan_counts[nan_counts > 0])
    else:
        print("No NaNs found in features.")

if __name__ == "__main__":
    inspect_evaluation_features()
