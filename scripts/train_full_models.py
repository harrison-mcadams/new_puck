
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

from puck import fit_xgs, fit_nested_xgs, fit_xgboost_nested, impute

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("TrainFull")

def main():
    logger.info("Starting Full Model Training (Skipping Download)...")
    
    # 1. Load Data
    # Look for all season CSVs
    data_dir = Path('data')
    season_files = list(data_dir.glob('**/*_df.csv'))
    
    # Also look for the top-level 20252026.csv if it exists
    current_season_csv = data_dir / "20252026.csv"
    if current_season_csv.exists() and current_season_csv not in season_files:
        season_files.append(current_season_csv)
    
    if not season_files:
        logger.error("No season files found! Did backfill download fail?")
        return
        
    logger.info(f"Found {len(season_files)} season files.")
    dfs = []
    for f in sorted(season_files):
        logger.info(f"Loading {f}...")
        try:
            df = pd.read_csv(f)
            
            # CRITICAL: Filter for regular season only
            # Regular season game IDs are e.g. 2023020001 (middle part is 02)
            if 'game_id' in df.columns:
                initial_len = len(df)
                # Ensure game_id is string for filtering
                df['game_id_str'] = df['game_id'].astype(str)
                # Regex for regular season (e.g., 2023020001)
                df = df[df['game_id_str'].str.contains(r'^\d{4}02\d{4}$')]
                df = df.drop(columns=['game_id_str'])
                filtered_len = len(df)
                if initial_len > filtered_len:
                    logger.info(f"  Filtered out {initial_len - filtered_len} non-regular season events.")
            
            dfs.append(df)
        except Exception as e:
            logger.error(f"Failed to load {f}: {e}")
            
    if not dfs:
        return
        
    full_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total rows (Regular Season Only): {len(full_df)}")
    
    # Check if 'event' column exists
    if 'event' not in full_df.columns:
        logger.error("Critical: 'event' column missing.")
        return

    # --- MODEL 1: SINGLE LAYER ---
    logger.info("\n--- Training Single Layer Model ---")
    
    # Filter out blocked shots
    df_single = full_df[full_df['event'] != 'blocked-shot'].copy()
    logger.info(f"Single Layer Training Rows: {len(df_single)}")
    
    features_single = ['distance', 'angle_deg', 'game_state', 'shot_type']
    
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
    # Uses 'point_pull' imputation
    logger.info("Applying 'point_pull' imputation...")
    df_nested_imp = impute.impute_blocked_shot_origins(full_df, method='point_pull')
    
    # XGBNestedXGClassifier handles the rest (encoding, categorical support, etc.)
    clf_nested = fit_xgboost_nested.XGBNestedXGClassifier(
        n_estimators=200,
        max_depth=6, # XGB standard
        use_calibration=True,
        use_balancing=True
    )
    
    logger.info("Fitting XGBoost Nested Model...")
    clf_nested.fit(df_nested_imp)
    
    # Save
    out_path_n = 'analysis/xgs/xg_model_nested.joblib'
    joblib.dump(clf_nested, out_path_n)

    # --- SANITY CHECK ---
    logger.info("--- STARTING SANITY CHECK PREDICTION ---")
    try:
        # Check training data sample
        sample = df_nested_imp.iloc[:5].copy()
        logger.info(f"Sanity Check Input Index: {sample.index}")
        probs = clf_nested.predict_proba(sample)
        logger.info(f"Sanity Check Probs (Train Data): \n{probs}")

        # Check synthetic data
        syn_df = pd.DataFrame([{
            'distance': 30, 'angle_deg': 0, 'game_state': '5v5', 
            'is_rebound': 0, 'is_rush': 0, 'shot_type': 'wrist',
            'time_elapsed_in_period_s': 600, 'period_number': 1, 'score_diff': 0,
            'last_event_type': 'Faceoff', 'last_event_time_diff': 10,
            'rebound_angle_change': 0, 'rebound_time_diff': 0, 'total_time_elapsed_s': 600,
            'shoots_catches': 'L',
            'event': 'shot-on-goal'
        }])
        # Ensure cols exist
        for f in clf_nested.features:
            if f not in syn_df.columns: syn_df[f] = 0
            
        logger.info(f"Sanity Check Synthetic Index: {syn_df.index}")
        probs_syn = clf_nested.predict_proba(syn_df)
        logger.info(f"Sanity Check Probs (Synthetic): \n{probs_syn}")
        logger.info("--- SANITY CHECK PASSED ---")

    except Exception as e:
        logger.error(f"--- SANITY CHECK FAILED: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Save Metadata (Nested handles its own features, but we can save generic info)
    # The NestedXGClassifier class stores feature lists in its config
    meta_n = {
        'model_type': 'nested',
        'imputation': 'point_pull'
    }
    with open(out_path_n + '.meta.json', 'w') as f:
        json.dump(meta_n, f, indent=2)
        
    logger.info(f"Saved Nested Model to {out_path_n}")
    
    logger.info("Training Complete.")

if __name__ == "__main__":
    main()
