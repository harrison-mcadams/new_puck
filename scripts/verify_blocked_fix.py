import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from puck import nhl_api, parse, data_pipeline

def verify_game(game_id):
    print(f"\n=== Verifying Game {game_id} ===")
    
    # 1. Fetch Raw Feed
    feed = nhl_api.get_game_feed(game_id)
    if not feed:
        print("  Failed to fetch feed.")
        return

    # 2. Raw Parse (PBP shape)
    df_raw = parse._game(feed)
    if df_raw.empty:
        print("  Failed to parse plays.")
        return

    # Filter for blocks to see raw state
    blocks_raw = df_raw[df_raw['event'].str.lower() == 'blocked-shot'].copy()
    if blocks_raw.empty:
        print("  No blocked shots found.")
        return
        
    print(f"  Found {len(blocks_raw)} blocked shots.")
    
    # 3. Preprocess (Apply Fix)
    # We turn off imputation to see the isolated effect of attribution swap.
    df_proc = data_pipeline.preprocess_features(df_raw, apply_imputation=False, verbose=True)
    
    # 4. Compare
    blocks_proc = df_proc[df_proc['event'].str.lower() == 'blocked-shot'].copy()
    
    # Get team IDs for context
    home_id = feed.get('homeTeam', {}).get('id')
    away_id = feed.get('awayTeam', {}).get('id')
    home_abb = feed.get('homeTeam', {}).get('abbrev') or feed.get('homeTeam', {}).get('name')
    away_abb = feed.get('awayTeam', {}).get('abbrev') or feed.get('awayTeam', {}).get('name')
    
    print(f"  Home: {home_abb} ({home_id}), Away: {away_abb} ({away_id})")
    
    for i in range(min(5, len(blocks_raw))):
        raw_row = blocks_raw.iloc[i]
        proc_row = blocks_proc.iloc[i]
        
        # In API raw, team_id is the BLOCKER (Defense).
        # In processed, team_id should be the SHOOTER (Offense).
        
        # Find which team the API owner is
        owner_is_home = (raw_row['team_id'] == home_id)
        owner_label = "HOME" if owner_is_home else "AWAY"
        
        new_is_home = (proc_row['team_id'] == home_id)
        new_label = "HOME" if new_is_home else "AWAY"
        
        print(f"  Block {i+1}:")
        print(f"    Raw Owner:  {raw_row['team_id']} ({owner_label}) | X: {raw_row['x']:.1f}")
        print(f"    Proc Owner: {proc_row['team_id']} ({new_label}) | X: {proc_row['x']:.1f} | Dist: {proc_row['distance']:.1f}")
        
        # Correctness check:
        # If Raw Owner was HOME, Proc Owner should be AWAY.
        if raw_row['team_id'] == proc_row['team_id']:
            print("    [FAIL] Team ID was not swapped!")
        else:
            print("    [PASS] Team ID successfully swapped to opponent.")
            
        # Orientation check:
        # After pipeline, all shots should be attacking RIGHT (X > 0).
        if proc_row['x'] < 0:
            print(f"    [WARN] Negative X after processing: {proc_row['x']:.1f}")
        else:
            print(f"    [PASS] Correctly oriented toward offensive goal (X > 0).")

# Test 202021
verify_game("2020020151")

# Test 202425
verify_game("2024020151")
