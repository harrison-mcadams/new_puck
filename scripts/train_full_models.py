
import sys
import os
import pandas as pd
import numpy as np
import joblib
import json
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from puck import fit_xgs, fit_nested_xgs, impute

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("TrainFull")

def main():
    logger.info("Starting Full Model Training (Skipping Download)...")
    
    # 1. Load Data
    # Look for all season CSVs
    data_dir = Path('data')
    season_files = list(data_dir.glob('**/*_df.csv'))
    
    if not season_files:
        logger.error("No season files found! Did backfill download fail?")
        return
        
    logger.info(f"Found {len(season_files)} season files.")
    dfs = []
    for f in sorted(season_files):
        logger.info(f"Loading {f}...")
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            logger.error(f"Failed to load {f}: {e}")
            
    if not dfs:
        return
        
    full_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total rows: {len(full_df)}")
    
    # Check if 'event' column exists
    if 'event' not in full_df.columns:
        logger.error("Critical: 'event' column missing.")
        return

    # --- MODEL 1: SINGLE LAYER ---
    logger.info("\n--- Training Single Layer Model ---")
    
    # Filter out blocked shots
    df_single = full_df[full_df['event'] != 'blocked-shot'].copy()
    logger.info(f"Single Layer Training Rows: {len(df_single)}")
    
    features_single = ['distance', 'angle_deg', 'game_state', 'is_net_empty', 'shot_type']
    
    # Clean/Preprocess
    df_single_mod, final_feats_s, cat_map_s = fit_xgs.clean_df_for_model(df_single, features_single)
    
    # Fit (max_depth=10 enforced)
    clf_single, final_feats_s, _ = fit_xgs.fit_model(
        df_single_mod, 
        feature_cols=final_feats_s, 
        n_estimators=200, 
        max_depth=10,
        progress=True
    )
    
    # Save
    out_path_s = 'analysis/xgs/xg_model_single.joblib'
    joblib.dump(clf_single.clf if isinstance(clf_single, fit_xgs.SingleXGClassifier) else clf_single, out_path_s)
    
    # Save Metadata
    # Helper for deep conversion
    def default_json(t):
        if isinstance(t, np.ndarray): return t.tolist()
        if isinstance(t, (np.int64, np.int32)): return int(t)
        if isinstance(t, (np.float64, np.float32)): return float(t)
        return str(t)

    meta_s = {
        'final_features': final_feats_s,
        'categorical_levels_map': cat_map_s,
        'model_type': 'single'
    }
    with open(out_path_s + '.meta.json', 'w') as f:
        json.dump(meta_s, f, indent=2, default=default_json)
    logger.info(f"Saved Single Layer to {out_path_s}")

    # --- MODEL 2: NESTED MODEL ---
    logger.info("\n--- Training Nested Model ---")
    # Uses 'mean_6' imputation
    logger.info("Applying 'mean_6' imputation...")
    df_nested_imp = impute.impute_blocked_shot_origins(full_df, method='mean_6')
    
    # NestedXGClassifier handles the rest (encoding, etc.)
    clf_nested = fit_nested_xgs.NestedXGClassifier(
        n_estimators=200,
        max_depth=10,
        prevent_overfitting=True
    )
    
    logger.info("Fitting Nested Model...")
    clf_nested.fit(df_nested_imp)
    
    # Save
    out_path_n = 'analysis/xgs/xg_model_nested.joblib'
    joblib.dump(clf_nested, out_path_n)
    
    # Save Metadata (Nested handles its own features, but we can save generic info)
    # The NestedXGClassifier class stores feature lists in its config
    meta_n = {
        'model_type': 'nested',
        'imputation': 'mean_6'
    }
    with open(out_path_n + '.meta.json', 'w') as f:
        json.dump(meta_n, f, indent=2)
        
    logger.info(f"Saved Nested Model to {out_path_n}")
    
    logger.info("Training Complete.")

if __name__ == "__main__":
    main()
