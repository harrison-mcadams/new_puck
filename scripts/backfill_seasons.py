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

def backfill():
    all_dfs = []
    
    # ensure output dir exists for model
    model_path = 'analysis/xgs/xg_model.joblib'
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    for season in SEASONS:
        print(f"\n==========================================")
        print(f" PROCESSING SEASON: {season}")
        print(f"==========================================\n")
        
        # 1. Regenerate Data
        # We store data in data/{season} to keep it organized
        out_dir = os.path.join('data', season)
        os.makedirs(out_dir, exist_ok=True)
        
        print(f"Downloading/Parsing data for {season}...")
        try:
            # use _scrape directly to control memory usage better
            # max_workers=2 to be gentle on Pi resources
            # return_feeds=False to avoid holding raw JSONs in RAM
            res = parse._scrape(
                season=season, 
                out_dir='data', 
                use_cache=True, 
                verbose=True,
                max_workers=2,         # <-- Reduced from default 8
                return_feeds=False,    # <-- Critical: Don't keep raw data in RAM
                return_elaborated_df=True,
                process_elaborated=True,
                save_raw=True,         # Save to disk is fine, but don't keep in RAM
                save_json=False,       # Skip the giant JSON file to save IO/Time
                save_csv=True          # We want the CSV backup
            )
            # When return_elaborated_df=True and return_feeds=False, 
            # and legacy is not triggered, _scrape returns the DataFrame directly.
            if isinstance(res, dict):
                 df = res.get('elaborated_df')
            else:
                 df = res
        except Exception as e:
            print(f"Error parsing season {season}: {e}")
            continue
        
        if df is None or df.empty:
            print(f"Warning: No data for {season}, skipping.")
            continue
            
        # Append to our master list
        all_dfs.append(df)
        print(f" -> Loaded {len(df)} rows. Total accumulated seasons: {len(all_dfs)}")
        
        # 2. Retrain Model on Cumulative Data
        print(f"\n>>> Retraining xG Model on ALL {len(all_dfs)} AVAILABLE SEASONS...")
        
        try:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            print(f"    Total training rows: {len(combined_df)}")
            
            # Prepare for training
            features = ['distance', 'angle_deg', 'game_state', 'is_net_empty']
            
            # Clean/Encode
            # We let clean_df_for_model derive levels from the combined data
            model_df, final_feats, cat_map = fit_xgs.clean_df_for_model(combined_df, features)
            
            # Fit
            # n_estimators=200 is default, maybe increase slightly for massive data? kept default for speed.
            clf, _, _ = fit_xgs.fit_model(model_df, feature_cols=final_feats, n_estimators=200)
            
            # Save Model
            print(f"    Saving updated model to {model_path}...")
            joblib.dump(clf, model_path)
            
            # Save Metadata (Critical for making predictions correctly later)
            meta_path = model_path + '.meta.json'
            meta = {'final_features': final_feats, 'categorical_levels_map': cat_map}
            with open(meta_path, 'w', encoding='utf-8') as fh:
                json.dump(meta, fh)
                
            print(f"    [SUCCESS] Model updated with data up to {season}")

        except Exception as e:
            print(f"    [ERROR] Failed to retrain model: {e}")

    print("\n\nBackfill Complete!")

if __name__ == "__main__":
    backfill()
