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
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from puck import parse, fit_xgs, fit_nested_xgs, nhl_api

# Define seasons to process (Reverse Chronological)
# From current season 20252026 back to 20152016 (10 seasons)
SEASONS = [f"{y}{y+1}" for y in range(2025, 2014, -1)]

def backfill():
    parser = argparse.ArgumentParser(description='Backfill NHL data.')
    parser.add_argument('--resume', action='store_true', help='Skip deletion of existing data to resume download.')
    args = parser.parse_args()

    print(f"Target Seasons: {SEASONS}")
    
    # --- PHASE 0: CLEANUP ---
    if not args.resume:
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
    else:
        print(f"\n==========================================")
        print(f" PHASE 0: SKIPPING CLEANUP (RESUME MODE)")
        print(f"==========================================\n")
        from puck.config import DATA_DIR
        data_dir = Path(DATA_DIR)
    
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
            games = nhl_api.get_season(season=season, team='all')
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

    print("\nBackfill Complete (Data Only). Models will be trained in the next step.")

if __name__ == "__main__":
    backfill()
