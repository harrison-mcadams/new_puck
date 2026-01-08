
import pandas as pd
import sys
import os
import json
import numpy as np

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import nhl_api
from puck import correction

SUMMARY_CSV = "analysis/blocked_shots_v2/blocked_shots_summary_batch.csv"

def debug():
    if not os.path.exists(SUMMARY_CSV):
        print("Summary CSV not found.")
        return

    df = pd.read_csv(SUMMARY_CSV)
    print(f"Loaded summary with {len(df)} rows.")
    
    # Pick a random row
    row = df.iloc[0]
    gid = row['game_id']
    bid = row['block_id']
    ox = row['x']
    oy = row['y']
    
    print(f"\nDEBUGGING Game {gid}, Block {bid}")
    print(f"Origin (Summary): x={ox}, y={oy}")
    
    # Load PBP
    pbp_json = nhl_api.get_game_feed(gid)
    plays = pbp_json.get('plays', [])
    
    # Find block
    block_play = None
    for p in plays:
        if p.get('eventId') == bid or str(p.get('eventId')) == str(bid):
            block_play = p
            break
            
    if not block_play:
        print("Block not found in PBP.")
        return
        
    details = block_play.get('details', {})
    raw_x = details.get('xCoord')
    raw_y = details.get('yCoord')
    owner = details.get('eventOwnerTeamId')
    
    print(f"Raw PBP Block: x={raw_x}, y={raw_y}, Owner={owner}")
    
    # Simulate Correction
    # Correction needs a DataFrame
    raw_blocks = [{
        'event': 'blocked-shot',
        'game_id': gid,
        'team_id': owner,
        'home_id': pbp_json.get('homeTeam', {}).get('id'),
        'away_id': pbp_json.get('awayTeam', {}).get('id'),
        'home_abb': pbp_json.get('homeTeam', {}).get('abbrev'), 
        'away_abb': pbp_json.get('awayTeam', {}).get('abbrev'),
        'home_team_defending_side': pbp_json.get('homeTeamDefendingSide'),
        'period': block_play.get('periodDescriptor', {}).get('number'),
        'x': float(raw_x) if raw_x else 0.0,
        'y': float(raw_y) if raw_y else 0.0,
        'event_id': int(bid)
    }]
    
    df_mini = pd.DataFrame(raw_blocks)
    print("\nBefore Correction:")
    print(df_mini[['x', 'y']].to_string(index=False))
    
    df_corr = correction.fix_blocked_shot_attribution(df_mini)
    
    bx_fixed = df_corr.iloc[0]['x']
    by_fixed = df_corr.iloc[0]['y']
    
    print("\nAfter Correction (fix_blocked_shot_attribution):")
    print(f"Block Fixed: x={bx_fixed}, y={by_fixed}")
    
    # Check Normalization Logic
    if bx_fixed < 0:
        bx_norm = -bx_fixed
        by_norm = -by_fixed
        ox_norm = -ox
        oy_norm = -oy
        flipped = True
    else:
        bx_norm = bx_fixed
        by_norm = by_fixed
        ox_norm = ox
        oy_norm = oy
        flipped = False
        
    print(f"\nNormalization (Flipped={flipped}):")
    print(f"Block Norm: {bx_norm}, {by_norm}")
    print(f"Origin Norm: {ox_norm}, {oy_norm}")
    
    # Check Vector
    dx = ox_norm - bx_norm
    print(f"Vector X (Origin - Block): {dx}")
    
    if abs(dx) > 50:
        print("ALERT: Huge X difference. Opposite ends?")
    else:
        print("Vector seems reasonable.")

if __name__ == "__main__":
    debug()
