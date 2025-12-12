import sys
import os
import shutil
import pandas as pd
import joblib
import json
import gc
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from puck import parse, fit_xgs, fit_nested_xgs

# Define seasons to process (Reverse Chronological)
# From current season 20252026 back to 20152016 (10 seasons)
SEASONS = [f"{y}{y+1}" for y in range(2025, 2014, -1)]

def backfill():
    print(f"Target Seasons: {SEASONS}")
    
    # --- PHASE 0: CLEANUP ---
    print(f"\n==========================================")
    print(f" PHASE 0: DELETING OLD DATA")
    print(f"==========================================\n")
    data_dir = Path('data')
    if data_dir.exists():
        # We want to be careful not to delete everything if not intended, 
        # but user said "delete the raw season data".
        # We'll delete standard season folders.
        for item in data_dir.iterdir():
            if item.is_dir() and item.name.isdigit():
                print(f"Deleting {item}...")
                shutil.rmtree(item)
    
    # --- PHASE 1: DOWNLOAD ---
    print(f"\n==========================================")
    print(f" PHASE 1: DOWNLOADING ALL SEASONS")
    print(f"==========================================\n")

    for season in SEASONS:
        out_dir = data_dir / season
        csv_path = out_dir / f"{season}_df.csv"
        
        # Check if already done (in case of re-run)
        if csv_path.exists() and csv_path.stat().st_size > 1000:
             print(f"Season {season} seems present ({csv_path}), skipping download.")
             continue

        print(f"Downloading/Parsing data for {season}...")
        try:
            parse._scrape(
                season=season, 
                out_dir='data', 
                use_cache=True, 
                verbose=True,
                max_workers=2,
                return_feeds=False,
                return_elaborated_df=False,
                process_elaborated=True,
                save_elaborated=True,
                save_raw=True, 
                save_json=False,
                save_csv=False
            )
            gc.collect()
        except Exception as e:
            print(f"Error parsing season {season}: {e}")
            continue

    # --- PHASE 2: TRAIN MODELS ---
    print(f"\n==========================================")
    print(f" PHASE 2: TRAINING MODELS")
    print(f"==========================================\n")
    
    all_dfs = []
    for season in SEASONS:
        csv_path = data_dir / season / f"{season}_df.csv"
        if csv_path.exists():
             print(f"Loading {season} from disk...")
             try:
                 df = pd.read_csv(csv_path)
                 all_dfs.append(df)
             except Exception as e:
                 print(f"Failed to load {csv_path}: {e}")
    
    if not all_dfs:
        print("No data found to train on!")
        return

    print(f"Concatenating {len(all_dfs)} seasons...")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    del all_dfs
    gc.collect()
    
    total_rows = len(combined_df)
    print(f"Total training rows (all events): {total_rows}")
    
    # ---------------------------------------------------------
    # MODEL 1: SINGLE LAYER (Standard)
    # ---------------------------------------------------------
    # Rules: No blocked shots. Max depth 10.
    print(f"\n--- Training Single Layer Model (No Blocks) ---")
    single_model_path = 'analysis/xgs/xg_model_single.joblib'
    
    try:
        # Filter handled inside fit_xgs.fit_model if we pass the right df?
        # Actually fit_xgs.clean_df_for_model does some filtering.
        # We will manually filter here to be explicit or rely on improved fit_xgs.
        # Let's rely on fit_xgs.clean_df_for_model having a new `exclude_blocked` param or we filter before.
        
        # We will assume fit_xgs update is done.
        # But to be safe, let's filter here too.
        
        features = ['distance', 'angle_deg', 'game_state', 'is_net_empty', 'shot_type']
        
        # We need a copy of combined_df since clean_df modifies? 
        # clean_df_for_model usually returns a new df.
        
        # Filter out blocked shots for single layer
        df_single = combined_df[combined_df['event'] != 'blocked-shot'].copy()
        
        model_df, final_feats, cat_map = fit_xgs.clean_df_for_model(df_single, features)
        
        clf_single, X_test, y_test = fit_xgs.fit_model(
            model_df, 
            feature_cols=final_feats, 
            n_estimators=200, 
            max_depth=10  # Enforce depth limit
        )
        
        joblib.dump(clf_single, single_model_path)
        
        # Metadata
        meta_path = single_model_path + '.meta.json'
        meta = {'final_features': final_feats, 'categorical_levels_map': cat_map, 'type': 'single_layer'}
        with open(meta_path, 'w', encoding='utf-8') as fh:
            json.dump(meta, fh)
            
        print(f"Saved Single Layer Model to {single_model_path}")
        
    except Exception as e:
        print(f"Failed Single Layer Training: {e}")

    # ---------------------------------------------------------
    # MODEL 2: NESTED MODEL (With Imputation)
    # ---------------------------------------------------------
    # Rules: Blocked shots allowed + Imputed. Max depth 10.
    print(f"\n--- Training Nested Model (With Blocks + Imputation) ---")
    nested_model_path = 'analysis/xgs/xg_model_nested.joblib' # Or directory
    
    try:
        # fit_nested_xgs.NestedXGClassifier handles its own internal fitting.
        # We just need to fit it.
        # But we need to preprocess for imputation FIRST? 
        # Or does the NestedXGClassifier handle it? 
        # Ideally, we pass the raw DF to it, but we need to run imputation first 
        # because the classifier might expect 'distance' to be correct.
        
        from puck.impute import impute_blocked_shot_origins
        
        # Apply imputation to a copy
        print("Applying 'mean_6' imputation...")
        df_nested_input = impute_blocked_shot_origins(combined_df, method='mean_6')
        
        clf_nested = fit_nested_xgs.NestedXGClassifier(
            n_estimators=200, 
            max_depth=10,  # Enforce depth limit
            prevent_overfitting=True
        )
        clf_nested.fit(df_nested_input)
        
        # We save the whole object
        joblib.dump(clf_nested, nested_model_path)
        print(f"Saved Nested Model to {nested_model_path}")
        
    except Exception as e:
        print(f"Failed Nested Training: {e}")
        import traceback
        traceback.print_exc()

    print("\nBackfill Complete!")

if __name__ == "__main__":
    backfill()

