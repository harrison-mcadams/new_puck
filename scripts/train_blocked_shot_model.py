
import os
import sys
import pandas as pd
import numpy as np
import json
import math

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import nhl_api
from puck import correction
from puck import config

SUMMARY_CSV = "analysis/blocked_shots/blocked_shots_summary_batch.csv"
MODEL_OUT = "puck/data/blocked_shot_model.json"

def main():
    if not os.path.exists(SUMMARY_CSV):
        print(f"Summary file not found: {SUMMARY_CSV}")
        return

    print(f"Loading summary: {SUMMARY_CSV}")
    df_sum = pd.read_csv(SUMMARY_CSV)
    
    # Filter by score
    if 'score' in df_sum.columns:
        initial_len = len(df_sum)
        df_sum = df_sum[df_sum['score'] > 0.4]
        print(f"Filtered from {initial_len} to {len(df_sum)} rows with score > 0.4")
    else:
        print("Warning: 'score' column not found in summary.")
    
    data_points = []
    
    # Group by game to optimize PBP loading
    games = df_sum['game_id'].unique()
    print(f"Processing {len(games)} games...")
    
    for gid in games:
        # Load PBP
        try:
             pbp_json = nhl_api.get_game_feed(gid)
             if not pbp_json:
                 continue
                 
             plays = pbp_json.get('plays', [])
             
             # Extract raw blocks
             raw_blocks = []
             for p in plays:
                 if p.get('typeDescKey') == 'blocked-shot':
                     eid = p.get('eventId')
                     details = p.get('details', {})
                     x = details.get('xCoord')
                     y = details.get('yCoord')
                     owner = details.get('eventOwnerTeamId')
                     
                     if x is not None and y is not None:
                         # Append dict matching correction.py expectation (roughly)
                         raw_blocks.append({
                             'event': 'blocked-shot',
                             'game_id': gid,
                             'team_id': owner,
                             'home_id': pbp_json.get('homeTeam', {}).get('id'),
                             'away_id': pbp_json.get('awayTeam', {}).get('id'),
                             'home_abb': pbp_json.get('homeTeam', {}).get('abbrev'), # correction uses this?
                             'away_abb': pbp_json.get('awayTeam', {}).get('abbrev'),
                             'home_team_defending_side': pbp_json.get('homeTeamDefendingSide'),
                             'period': p.get('periodDescriptor', {}).get('number'),
                             'x': float(x),
                             'y': float(y),
                             'event_id': int(eid) if eid else None
                         })
                         
             if not raw_blocks:
                 continue
                 
             df_blocks = pd.DataFrame(raw_blocks)
             
             # Apply Correction (to get Block X/Y in Shooter Perspective)
             df_corr = correction.fix_blocked_shot_attribution(df_blocks)
             
             # Match with Summary Data
             subset_sum = df_sum[df_sum['game_id'] == gid]
             
             for _, row in subset_sum.iterrows():
                 bid = row['block_id']
                 match = df_corr[df_corr['event_id'] == bid]
                 if match.empty:
                     match = df_corr[df_corr['event_id'].astype(str) == str(bid)]
                     
                 if not match.empty:
                     block_row = match.iloc[0]
                     
                     bx_fixed = block_row['x']
                     by_fixed = block_row['y']
                     
                     # Summary 'x'/'y' are True Origin
                     origin_x = row['x']
                     origin_y = row['y']
                     
                     # Frame Alignment Check
                     if abs(bx_fixed) > 25 and abs(origin_x) > 25:
                         if (bx_fixed > 0 and origin_x < 0) or (bx_fixed < 0 and origin_x > 0):
                             # Flip Origin to match Block
                             origin_x = -origin_x
                             origin_y = -origin_y

                     # Normalize to Attack Right (X > 0)
                     if bx_fixed < 0:
                         bx_norm, by_norm = -bx_fixed, -by_fixed
                         ox_norm, oy_norm = -origin_x, -origin_y
                     else:
                         bx_norm, by_norm = bx_fixed, by_fixed
                         ox_norm, oy_norm = origin_x, origin_y
                         
                     # Sanity Check: Is Origin "Backwards"?
                     d_block = math.hypot(bx_norm - 89, by_norm)
                     d_origin = math.hypot(ox_norm - 89, oy_norm)
                     
                     if d_origin < (d_block - 5.0): # 5ft buffer
                         continue

                     data_points.append({
                         'bx': bx_norm, 'by': by_norm,
                         'ox': ox_norm, 'oy': oy_norm
                     })
                     
        except Exception as e:
            # print(f"Error processing game {gid}: {e}")
            continue

    print(f"Collected {len(data_points)} matched data points.")
    
    # Create Model (Binning)
    df_data = pd.DataFrame(data_points)
    
    # Bin size 5ft
    BIN_SIZE = 5.0
    
    # Grid Coverage
    x_bins = np.arange(0, 105, BIN_SIZE)
    y_bins = np.arange(-50, 55, BIN_SIZE) 
    
    df_data['x_bin'] = pd.cut(df_data['bx'], bins=x_bins, labels=False)
    df_data['y_bin'] = pd.cut(df_data['by'], bins=y_bins, labels=False)
    
    # First Pass: Raw Sums
    raw_grid = {}
    grouped = df_data.groupby(['x_bin', 'y_bin'])
    
    for (xb, yb), group in grouped:
        raw_grid[(xb, yb)] = {
            'sum_ox': group['ox'].sum(),
            'sum_oy': group['oy'].sum(),
            'count': len(group)
        }
        
    # Second Pass: Smoothing
    model = {}
    
    max_x_bin = len(x_bins) - 1
    max_y_bin = len(y_bins) - 1
    
    for i in range(max_x_bin):
        for j in range(max_y_bin):
            
            # Neighborhood
            w_sum_ox = 0
            w_sum_oy = 0
            w_count = 0
            
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    ni, nj = i + di, j + dj
                    if (ni, nj) in raw_grid:
                        cell = raw_grid[(ni, nj)]
                        weight = 2.0 if (di==0 and dj==0) else 1.0
                        
                        w_sum_ox += cell['sum_ox'] * weight 
                        w_sum_oy += cell['sum_oy'] * weight 
                        w_count += cell['count'] * weight
            
            if w_count > 0:
                mx = w_sum_ox / w_count
                my = w_sum_oy / w_count
                real_n = raw_grid.get((i,j), {}).get('count', 0)
                
                # Threshold for saving bin
                # Lowered from 2.0 to 1.0 to include more bins, but heavily damped
                if w_count >= 1.0:
                    
                    # DAMPING / "Do No Harm"
                    # If we have low confidence (low w_n), shrinking the "adjustment" 
                    # vector towards 0 (meaning Imputed Origin = Block Location).
                    # This ensures that in sparse areas we don't make wild guesses based on 1-2 points.
                    
                    bx_center = i * BIN_SIZE + BIN_SIZE/2
                    by_center = j * BIN_SIZE - 50.0 + BIN_SIZE/2
                    
                    # Adjustment Vector
                    adj_x = mx - bx_center
                    adj_y = my - by_center
                    
                    # Damping Factor
                    # Full trust at N >= 10?
                    # Linear ramp: 0.0 at N=0, 1.0 at N=10
                    N_TRUST = 10.0
                    damping = min(w_count, N_TRUST) / N_TRUST
                    
                    final_mx = bx_center + (adj_x * damping)
                    final_my = by_center + (adj_y * damping)
                    
                    k = f"{int(i)}_{int(j)}"
                    model[k] = {
                        'mx': float(final_mx),
                        'my': float(final_my),
                        'n': int(real_n),
                        'w_n': float(w_count),
                        'damping': float(damping)
                    }
        
    print(f"Model trained with {len(model)} bins using {len(data_points)} observations.")
    
    model_struct = {
        'meta': {
            'bin_size': BIN_SIZE,
            'x_min': 0, 'x_max': 100,
            'y_min': -50, 'y_max': 50
        },
        'bins': model
    }
    
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    with open(MODEL_OUT, 'w') as f:
        json.dump(model_struct, f, indent=2)
        
    print(f"Saved model to {MODEL_OUT}")

if __name__ == "__main__":
    main()
