import pandas as pd
import numpy as np
import os
import sys
import json
import logging
import traceback

# Ensure we can import from puck
sys.path.append(os.getcwd())
try:
    from puck import data_pipeline
    from puck import nhl_api
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def verify_game(game_id: str, season_label: str):
    print(f"\n========================================================")
    print(f"VERIFYING SEASON: {season_label} | GAME: {game_id}")
    print(f"========================================================")
    
    try:
        # 1. Fetch raw data
        feed = nhl_api.get_game_feed(int(game_id))
        if not feed:
            print(f"Failed to fetch game feed for {game_id}.")
            return
        
        raw_events = feed.get('plays', [])

        # 2. Find a blocked shot in raw data
        raw_blocks = [e for e in raw_events if e.get('typeDescKey') == 'blocked-shot' or str(e.get('typeCode')) == '508']
            
        if not raw_blocks:
            print(f"No blocked shots found in raw data for game {game_id}.")
            return

        # Take the first one as a sample
        sample_raw = raw_blocks[0]
        details = sample_raw.get('details', {})
        
        # Map raw fields to what the pipeline expects
        df_raw = pd.DataFrame([sample_raw])
        df_raw['game_id'] = game_id
        df_raw['event'] = sample_raw.get('typeDescKey', 'blocked-shot')
        df_raw['team_id'] = details.get('eventOwnerTeamId')
        df_raw['home_id'] = feed.get('homeTeam', {}).get('id')
        df_raw['away_id'] = feed.get('awayTeam', {}).get('id')
        df_raw['x'] = details.get('xCoord')
        df_raw['y'] = details.get('yCoord')
        
        # Temporal fields for HTML join
        df_raw['period'] = sample_raw.get('periodDescriptor', {}).get('number')
        df_raw['time_in_period'] = sample_raw.get('timeInPeriod')
        
        # Defending side
        df_raw['home_team_defending_side'] = sample_raw.get('homeTeamDefendingSide', 'Unknown')
        
        raw_team_id = df_raw.iloc[0]['team_id']
        raw_shooter_id = details.get('shooterPlayerId')
        raw_blocker_id = details.get('blockingPlayerId')
        raw_shot_type = details.get('shotType', 'None')
        
        print(f"\n[RAW API DATA]")
        print(f"  Event Type:   {df_raw.iloc[0]['event']} ({sample_raw.get('typeCode')})")
        print(f"  Owner Team:   {raw_team_id} (Blocker's Team)")
        print(f"  Shooter ID:   {raw_shooter_id}")
        print(f"  Blocker ID:   {raw_blocker_id}")
        print(f"  Raw ShotType: {raw_shot_type}")
        print(f"  Raw Coords:   ({df_raw.iloc[0]['x']}, {df_raw.iloc[0]['y']})")
        print(f"  Temporal:     P{df_raw.iloc[0]['period']} {df_raw.iloc[0]['time_in_period']}")

        # 3. Process through data_pipeline
        print("\n[PIPELINE STARTING...]")
        df_processed = data_pipeline.preprocess_features(
            df_raw, 
            game_id=game_id, 
            apply_attribution_fix=True, 
            apply_html_enrichment=True,
            apply_filtering=False,
            apply_imputation=False,
            verbose=True
        )
        
        if df_processed.empty:
            print("\n[PIPELINE] Empty result.")
        else:
            row = df_processed.iloc[0]
            print(f"\n[PIPELINE PROCESSED]")
            print(f"  Final Team ID: {row['team_id']} (Attacker: {'YES' if row['team_id'] != raw_team_id else 'NO'})")
            print(f"  Final Coords:  ({row['x']:.2f}, {row['y']:.2f})")
            print(f"  Final ShotTyp: {row.get('shot_type', 'N/A')}")
            print(f"  Distance:      {row['distance']:.2f}")
            print(f"  Angle:         {row['angle_deg']:.2f}")

    except Exception:
        print(f"\n[ERROR] Season {season_label} | Game {game_id}")
        traceback.print_exc()

if __name__ == "__main__":
    # Sample games for modern era seasons
    samples = [
        ("2020020004", "2020-21"),
        ("2021020001", "2021-22"),
        ("2022020002", "2022-23"),
        ("2023020003", "2023-24"),
        ("2024020005", "2024-25")
    ]
    
    for gid, label in samples:
        verify_game(gid, label)
