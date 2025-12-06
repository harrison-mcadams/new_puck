import sys
import os
import pandas as pd
import joblib
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from puck import parse, fit_xgs

# Define seasons to process (Reverse Chronological)
# From current season 20252026 back to 20142015
SEASONS = [f"{y}{y+1}" for y in range(2025, 2013, -1)]

import gc

def backfill():
    # Phase 1: Ensure all data is downloaded/processed to disk
    # We do NOT keep DFs in memory here to avoid OOM
    print(f"\n==========================================")
    print(f" PHASE 1: DOWNLOADING ALL SEASONS")
    print(f"==========================================\n")

    for season in SEASONS:
        out_dir = os.path.join('data', season)
        csv_path = os.path.join(out_dir, f"{season}.csv")
        
        # Check if already done to save time
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 1000:
             print(f"Season {season} seems present ({csv_path}), skipping download.")
             continue

        print(f"Downloading/Parsing data for {season}...")
        try:
            # We use save_csv=True, so it writes to disk.
            # We ignore the return value to free memory immediately.
            parse._scrape(
                season=season, 
                out_dir='data', 
                use_cache=True, 
                verbose=True,
                max_workers=2,
                return_feeds=False,
                return_elaborated_df=False, # Don't return it, just save it!
                process_elaborated=True,
                save_raw=True, 
                save_json=False,
                save_csv=True
            )
            # Force cleanup
            gc.collect()
        except Exception as e:
            print(f"Error parsing season {season}: {e}")
            continue

    # Phase 2: Train once on all available data
    print(f"\n==========================================")
    print(f" PHASE 2: TRAINING MODEL (ONE-SHOT)")
    print(f"==========================================\n")
    
    model_path = 'analysis/xgs/xg_model.joblib'
    all_dfs = []

    for season in SEASONS:
        csv_path = os.path.join('data', season, f"{season}.csv")
        if os.path.exists(csv_path):
             print(f"Loading {season} from disk...")
             try:
                 # Load only necessary cols to save RAM if possible?
                 # fit_xgs handles loading, but we are aggregating.
                 # Let's load full for now, but watch RAM.
                 df = pd.read_csv(csv_path)
                 all_dfs.append(df)
             except Exception as e:
                 print(f"Failed to load {csv_path}: {e}")
    
    if not all_dfs:
        print("No data found to train on!")
        return

    print(f"Concatenating {len(all_dfs)} seasons...")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Free the list of individual DFs immediately
    del all_dfs
    gc.collect()
    
    print(f"Total training rows: {len(combined_df)}")
    print("Training Random Forest...")
    
    try:
        features = ['distance', 'angle_deg', 'game_state', 'is_net_empty']
        model_df, final_feats, cat_map = fit_xgs.clean_df_for_model(combined_df, features)
        
        # Free original combined_df
        del combined_df
        gc.collect()
        
        clf, _, _ = fit_xgs.fit_model(model_df, feature_cols=final_feats, n_estimators=200)
        
        print(f"Saving model to {model_path}...")
        joblib.dump(clf, model_path)
        
        meta_path = model_path + '.meta.json'
        meta = {'final_features': final_feats, 'categorical_levels_map': cat_map}
        with open(meta_path, 'w', encoding='utf-8') as fh:
            json.dump(meta, fh)
            
        print("[SUCCESS] Model trained and saved.")
        
    except Exception as e:
        print(f"[ERROR] Training failed (probably OOM): {e}")

    print("\nBackfill Complete!")

if __name__ == "__main__":
    backfill()
