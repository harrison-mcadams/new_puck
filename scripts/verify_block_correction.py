
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from puck import parse, nhl_api, data_pipeline
import importlib
importlib.reload(data_pipeline)

def verify_blocked_shots():
    # 1. Fetch a known game with blocked shots (e.g. 2025020008, CHI vs ...)
    game_id = 2025020008 
    season = "20252026"
    print(f"Fetching raw feed for {game_id}...")
    try:
        raw_feed = nhl_api.get_game_feed(game_id)
    except Exception as e:
        print(f"Failed to fetch feed: {e}")
        return

    # 2. Parse Raw Events
    print("Parsing events...")
    df_parsed = parse._game(raw_feed)
    
    if df_parsed.empty:
        print("Parsed DataFrame is empty.")
        return

    # Identify a blocked shot to track
    # We want an Offensive Zone block (Shooter in Offensive Zone).
    # If Home Defends Left (-89), Block near -60 is Offensive (for Away).
    # If Home Defends Right (+89), Block near +60 is Offensive (for Away).
    
    # Let's verify 'home_team_defending_side' first
    side = df_parsed['home_team_defending_side'].iloc[0]
    home_id = df_parsed['home_id'].dropna().unique()[0]
    
    # Define "Normal Block" definition:
    # 1. If Blocker is Home (Team == Home), and Side=Left (-89) -> Block X < -25.
    # 2. If Blocker is Away (Team != Home), and Side=Left (-89) -> Block X > 25. (Away defends Right).
    # vice versa for Side=Right.
    
    # We will iterate through blocks until we find one that matches.
    blocks = df_parsed[df_parsed['event'] == 'blocked-shot']
    
    target_block = None
    
    print(f"Searching {len(blocks)} blocks for a Normal Def-Zone Block...")
    
    for idx, row in blocks.iterrows():
        is_home_blocker = (row['team_id'] == home_id)
        x = row['x']
        
        # Side=Left: Home Def -89, Away Def +89.
        if side == 'left':
            if is_home_blocker:
                # Home Def Zone is Negative
                if x < -25:
                    target_block = row
                    print(f"Found Home Block in Def Zone (X={x})")
                    break
            else: # Away Blocker
                # Away Def Zone is Positive
                if x > 25:
                    target_block = row
                    print(f"Found Away Block in Def Zone (X={x})")
                    break
        else: # side=Right: Home Def +89, Away Def -89.
            if is_home_blocker:
                 # Home Def Zone is Positive
                 if x > 25:
                    target_block = row
                    print(f"Found Home Block in Def Zone (X={x})")
                    break
            else: # Away Blocker
                 # Away Def Zone is Negative
                 if x < -25:
                    target_block = row
                    print(f"Found Away Block in Def Zone (X={x})")
                    break
                    
    if target_block is None:
        print("Could not find a Normal Def-Zone block. Using first block as fallback.")
        target_block = blocks.iloc[0]

    idx = target_block.name
    idx = target_block.name
    
    print("\n--- BEFORE PROCESSING (Parsed Only) ---")
    print(target_block[['event', 'team_id', 'home_id', 'away_id', 'x', 'y', 'home_team_defending_side']].to_dict())
    
    # 3. Run Pipeline
    print("\nRunning preprocess_features...")
    # We must ensure 'home_team_defending_side' is present (it is in parsed).
    
    df_processed = data_pipeline.preprocess_features(
        df_parsed, 
        is_training=False, 
        apply_arena_adjustments=False, # simpliy
        apply_imputation=False # simplify
    )
    
    processed_block = df_processed.iloc[idx] # Assuming index preservation
    
    print("\n--- AFTER PROCESSING ---")
    print(processed_block[['event', 'team_id', 'x', 'y', 'distance', 'angle_deg']].to_dict())
    
    # Validation Logic
    # 1. Did team_id swap?
    raw_team = target_block['team_id']
    proc_team = processed_block['team_id']
    
    if raw_team != proc_team:
        print(f"\nSUCCESS: Team ID swapped from {raw_team} to {proc_team}.")
    else:
        print(f"\nFAILURE: Team ID did NOT swap. Remains {raw_team}.")
        
    # 2. Coordinate Check
    # If swapped (to Shooter), coordinate should be Offensive Zone (usually positive X after standardization, or close to net)
    # If parsed X was -61 (Defensive)
    # And we oriented to Shooter (who is Attacking Right).
    # Then X should be 61 (Offensive).
    
    raw_x = target_block['x']
    proc_x = processed_block.get('x_adj', processed_block['x'])
    
    print(f"Raw X: {raw_x}")
    print(f"Processed X: {proc_x}")
    
    if abs(proc_x) > 25 and (np.sign(proc_x) != np.sign(raw_x)):
         print("Observation: Coordinate flipped sign (implies orientation change).")
    
    print(f"Final Distance: {processed_block['distance']}")

if __name__ == "__main__":
    verify_blocked_shots()
