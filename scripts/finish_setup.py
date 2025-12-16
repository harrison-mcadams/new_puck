import sys
import os
import shutil
import pandas as pd
import joblib
import json
import gc
from pathlib import Path
import time
import subprocess

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from puck import fit_xgs, fit_nested_xgs, config

def finish_setup():
    print("########################################################")
    print("###           RESUMING SETUP (SKIP SCRAPING)         ###")
    print("########################################################")
    
    # Use config paths to respect external drive if set
    DATA_DIR = Path(config.DATA_DIR)
    ANALYSIS_DIR = Path(config.ANALYSIS_DIR)
    
    print(f"Using Data Dir: {DATA_DIR}")
    print(f"Using Analysis Dir: {ANALYSIS_DIR}")

    # --- PHASE 2: TRAINING MODELS (Copied from backfill_seasons.py) ---
    print(f"\n==========================================")
    print(f" PHASE 2: TRAINING MODELS (From Existing Data)")
    print(f"==========================================\n")
    
    # Scan for available season files
    all_dfs = []
    # Identify seasons 2015-2026 roughly
    potential_seasons = [f"{y}{y+1}" for y in range(2025, 2014, -1)]
    
    found_seasons = []
    
    for season in potential_seasons:
        # Check standard location
        csv_path = DATA_DIR / season / f"{season}_df.csv"
        # Also check root data dir just in case
        csv_path_root = DATA_DIR / f"{season}.csv"
        
        target_path = None
        if csv_path.exists():
            target_path = csv_path
        elif csv_path_root.exists():
            target_path = csv_path_root
            
        if target_path:
             print(f"Loading {season} from {target_path}...")
             try:
                 df = pd.read_csv(target_path)
                 all_dfs.append(df)
                 found_seasons.append(season)
             except Exception as e:
                 print(f"Failed to load {target_path}: {e}")
                 
    if not all_dfs:
        print("No data found in data/ to train on! Did the download fail completely?")
        sys.exit(1)

    print(f"Concatenating {len(all_dfs)} seasons: {found_seasons}")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    del all_dfs
    gc.collect()
    
    total_rows = len(combined_df)
    print(f"Total training rows (all events): {total_rows}")
    
    # ---------------------------------------------------------
    # MODEL 1: SINGLE LAYER (Standard)
    # ---------------------------------------------------------
    print(f"\n--- Training Single Layer Model (No Blocks) ---")
    single_model_path = ANALYSIS_DIR / 'xgs' / 'xg_model_single.joblib'
    os.makedirs(os.path.dirname(single_model_path), exist_ok=True)
    
    try:
        features = ['distance', 'angle_deg', 'game_state', 'is_net_empty', 'shot_type']
        df_single = combined_df[combined_df['event'] != 'blocked-shot'].copy()
        model_df, final_feats, cat_map = fit_xgs.clean_df_for_model(df_single, features)
        
        clf_single, X_test, y_test = fit_xgs.fit_model(
            model_df, 
            feature_cols=final_feats, 
            n_estimators=200, 
            max_depth=10 
        )
        
        joblib.dump(clf_single, single_model_path)
        
        # Metadata
        meta_path = str(single_model_path) + '.meta.json'
        meta = {'final_features': final_feats, 'categorical_levels_map': cat_map, 'type': 'single_layer'}
        with open(meta_path, 'w', encoding='utf-8') as fh:
            json.dump(meta, fh)
            
        print(f"Saved Single Layer Model to {single_model_path}")
        
    except Exception as e:
        print(f"Failed Single Layer Training: {e}")

    # ---------------------------------------------------------
    # MODEL 2: NESTED MODEL (With Imputation)
    # ---------------------------------------------------------
    print(f"\n--- Training Nested Model (With Blocks + Imputation) ---")
    nested_model_path = ANALYSIS_DIR / 'xgs' / 'xg_model_nested.joblib'
    os.makedirs(os.path.dirname(nested_model_path), exist_ok=True)
    
    try:
        from puck.impute import impute_blocked_shot_origins
        print("Applying 'mean_6' imputation...")
        # Note: impute_blocked_shot_origins modifies a copy if not inplace? 
        # Actually it usually returns a new DF or modifies. Let's assume standard usage.
        df_nested_input = impute_blocked_shot_origins(combined_df, method='mean_6')
        
        clf_nested = fit_nested_xgs.NestedXGClassifier(
            n_estimators=200, 
            max_depth=10, 
            prevent_overfitting=True
        )
        clf_nested.fit(df_nested_input)
        
        joblib.dump(clf_nested, nested_model_path)
        print(f"Saved Nested Model to {nested_model_path}")
        
    except Exception as e:
        print(f"Failed Nested Training: {e}")
        import traceback
        traceback.print_exc()

    print("\nTraining Complete!")
    
    # FREE MEMORY before Daily
    del combined_df
    try: del df_single
    except: pass
    try: del df_nested_input
    except: pass
    gc.collect()

    # --- PHASE 3: DAILY UPDATE ---
    print(f"\n==========================================")
    print(f" PHASE 3: RUNNING DAILY UPDATE (Forced)")
    print(f"==========================================\n")
    
    current_season = "20252026"
    print(f"Running daily.py for {current_season} with --force...")
    
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'daily.py')
    
    try:
        subprocess.run(
            [sys.executable, script_path, '--season', current_season, '--force'], 
            check=True
        )
        print("\nSUCCESS: Setup Finished!")
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: daily.py failed with exit code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    finish_setup()
