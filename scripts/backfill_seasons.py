import sys
import os
import shutil
import pandas as pd
import joblib
import json
import gc
from pathlib import Path
import time
import random

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from puck import parse, fit_xgs, fit_nested_xgs, nhl_api

# Define seasons to process (Reverse Chronological)
# From current season 20252026 back to 20152016 (10 seasons)
SEASONS = [f"{y}{y+1}" for y in range(2025, 2014, -1)]

def backfill():
    print(f"Target Seasons: {SEASONS}")
    
    # --- PHASE 0: CLEANUP ---
    print(f"\n==========================================")
    print(f" PHASE 0: DELETING OLD DATA")
    print(f"==========================================\n")
    from puck.config import DATA_DIR, ANALYSIS_DIR
    data_dir = Path(DATA_DIR)
    if data_dir.exists():
        # We want to be careful not to delete everything if not intended, 
        # but user said "delete the raw season data".
        # We'll delete standard season folders.
        for item in data_dir.iterdir():
            if item.is_dir() and item.name.isdigit():
                print(f"Deleting {item}...")
                shutil.rmtree(item)
    
    # --- PHASE 1: DOWNLOAD & PARSE (Pipeline with Manual Memory Management) ---
    print(f"\n==========================================")
    print(f" PHASE 1: DOWNLOADING & PARSING (Pi Optimized)")
    print(f"==========================================\n")

    for season in SEASONS:
        out_dir = data_dir / season
        csv_path = out_dir / f"{season}_df.csv"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already done (in case of re-run)
        if csv_path.exists() and csv_path.stat().st_size > 1000:
             print(f"Season {season} seems present ({csv_path}), skipping download.")
             continue

        print(f"Fetching games list for {season}...")
        try:
            games = nhl_api.get_season(season=season)
        except Exception as e:
            print(f"Failed to get schedule for {season}: {e}")
            continue
            
        print(f"Found {len(games)} games for {season}. Processing sequentially...")
        
        season_records = []
        
        for i, gm in enumerate(games):
            gid = gm.get('id') or gm.get('gamePk') or gm.get('gameID')
            if not gid:
                continue
                
            try:
                # 1. Fetch
                feed = nhl_api.get_game_feed(gid)
                if not feed:
                    print(f"  [Warn] Empty feed for {gid}")
                    continue
                
                # 2. Save Raw (Individual JSON)
                # We save this so we don't have to re-download if we crash,
                # even though we aren't using the 'use_cache' logic of _scrape here 
                # to keep things simple.
                game_json_path = out_dir / f"game_{gid}.json"
                with game_json_path.open('w', encoding='utf-8') as f:
                    json.dump(feed, f, ensure_ascii=False)
                    
                # 3. Parse
                ev_df = parse._game(feed)
                
                # 4. Elaborate
                if ev_df is not None and not ev_df.empty:
                    edf = parse._elaborate(ev_df)
                    if edf is not None and not edf.empty:
                        # Append dicts to list (lighter than concating DFs repeatedly)
                        season_records.extend(edf.to_dict('records'))
                        
                # 5. MEMORY CLEANUP
                del feed
                del ev_df
                # del edf  # edf scope is local, but good practice
                
                if i % 50 == 0:
                    print(f"  Progress: {i}/{len(games)} games processed...")
                    gc.collect()
                    
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                print(f"  [Error] Failed game {gid}: {e}")
                continue

        # Save compacted season DF
        if season_records:
            print(f"Saving {len(season_records)} events for {season}...")
            df = pd.DataFrame.from_records(season_records)
            df.to_csv(csv_path, index=False)
            del df
            del season_records
            gc.collect()
        else:
            print(f"Warning: No events found for {season}")

        gc.collect()

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
    single_model_path = os.path.join(ANALYSIS_DIR, 'xgs', 'xg_model_single.joblib')
    
    try:
        features = ['distance', 'angle_deg', 'game_state', 'is_net_empty', 'shot_type']
        
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
    nested_model_path = os.path.join(ANALYSIS_DIR, 'xgs', 'xg_model_nested.joblib')
    
    try:
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
