
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
    
    # Prepare Role Column globally
    if 'shooter_role' in df_sum.columns:
        df_sum['role'] = df_sum['shooter_role'].fillna('F')
    else:
        df_sum['role'] = None

    bios_cache = {}

    for gid in games:
        # Determine season string from game_id (e.g. 202302... -> 20232024)
        s_str = str(gid)[:4]
        season_full = f"{s_str}{int(s_str)+1}"

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
                     shooter_id = details.get('shootingPlayerId') # NEW
                     
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
                             'event_id': int(eid) if eid else None,
                             'shooter_id': shooter_id
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
                     
                     # Determine Role
                     role = 'F' # Default
                     # 1. Use Summary Role (Fastest)
                     if row.get('role'):
                         role = row['role']
                     # 2. Use Block/PBP Shooter ID (Slower fallback)
                     elif block_row.get('shooter_id'):
                         shooter_id_val = block_row['shooter_id']
                         # Lazy load bios only if needed
                         if season_full not in bios_cache:
                             print(f"Fetching bios for {season_full}...")
                             bios_cache[season_full] = nhl_api.get_season_player_bios(season_full)
                         
                         current_bios = bios_cache[season_full]
                         b = current_bios.get(shooter_id_val) or current_bios.get(str(shooter_id_val)) or current_bios.get(int(shooter_id_val))
                         if b:
                             pos = b.get('positionCode')
                             if pos == 'D':
                                 role = 'D'
                     
                     # Summary 'x'/'y' are True Origin
                     origin_x = row['x']
                     origin_y = row['y']
                     
                     # Frame Alignment Check
                     # If Origin is "flipped" relative to block?
                     # Block X should generally be same sign as Origin X?
                     # Not necessarily if near center line.
                     # But usually both are in attacking zone (positive X for fixed, negative X for raw?)
                     # correction.py returns "Fixed" coords (Shooter Perspective, Net at +89)
                     
                     # Wait, correction.py returns coords where Net is at +/- 89?
                     # Actually correction.py normalizes so attacking team shoots towards positive X?
                     # Let's assume bx_fixed satisfies "Attacking Zone is +X" or similar.
                     # Summary data (Edge) is raw rink coords.
                     
                     # Heuristic: If Block is clearly on one side, Origin should be on same side.
                     # Edge X: -100 to 100.
                     # Block X (fixed): ?
                     
                     # Let's simplisticly assume if they are far apart in sign, flip Origin.
                     if abs(bx_fixed) > 25 and abs(origin_x) > 25:
                         if (bx_fixed > 0 and origin_x < 0) or (bx_fixed < 0 and origin_x > 0):
                             origin_x = -origin_x
                             origin_y = -origin_y # Rotate 180
                             
                     # Normalize to "Right Attack" (Shooter at X < 89, shooting at 89?)
                     # Or "Standard Rink" (0..100)
                     # Let's standardize on: Net at +89.
                     # If bx_fixed is negative (defending zone blocks are usually in defensive zone...)
                     # Wait. `correction.fix_blocked_shot_attribution` normalizes coords so they are "Team Shooting Perspective"?
                     # Usually that means Attacking Zone is X > 0.
                     # Blocked shots usually happen in Defensive zone of the BLOCKER, which is Offensive Zone of SHOOTER.
                     # So bx_fixed > 0 is expected.
                     
                     if bx_fixed < 0:
                         bx_norm, by_norm = -bx_fixed, -by_fixed
                         ox_norm, oy_norm = -origin_x, -origin_y
                     else:
                         bx_norm, by_norm = bx_fixed, by_fixed
                         ox_norm, oy_norm = origin_x, origin_y
                         
                     # Sanity Check: Is Origin FURTHER than Block?
                     # Net is at 89.
                     # Distance from Net:
                     d_block = math.hypot(bx_norm - 89, by_norm)
                     d_origin = math.hypot(ox_norm - 89, oy_norm)
                     
                     # Origin should be further from net than Block.
                     # Origin should be further from net than Block.
                     if d_origin < d_block: # Strict physics check (no buffer)
                         continue
 
                     data_points.append({
                         'bx': bx_norm, 'by': by_norm,
                         'ox': ox_norm, 'oy': oy_norm,
                         'role': role
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

    # Split by Role (F vs D)
    # We will train two models: 'F' and 'D'.
    # If role is unknown, maybe we skip or put in F? Let's skip unknown for purity.
    
    unique_roles = ['F', 'D']
    
    for role in unique_roles:
        print(f"\n--- Training Model for Role: {role} ---")
        
        subset = df_data[df_data['role'] == role].copy()
        if subset.empty:
            print(f"No data for role {role}!")
            continue
            
        print(f"Data points: {len(subset)}")

        # First Pass: Raw Sums
        raw_grid = {}
        grouped = subset.groupby(['x_bin', 'y_bin'])
        
        for (xb, yb), group in grouped:
            raw_grid[(xb, yb)] = {
                'sum_ox': group['ox'].sum(),
                'sum_oy': group['oy'].sum(),
                'sum_sq_ox': (group['ox']**2).sum(),
                'sum_sq_oy': (group['oy']**2).sum(),
                'count': len(group)
            }
            
        # Second Pass: Smoothing
        model = {}
        
        max_x_bin = len(x_bins) - 1
        max_y_bin = len(y_bins) - 1
        
        valid_bins_count = 0
        
        for i in range(max_x_bin):
            for j in range(max_y_bin):
                
                # Neighborhood
                w_sum_ox = 0
                w_sum_oy = 0
                w_sum_sq_ox = 0
                w_sum_sq_oy = 0
                w_count = 0
                
                for di in range(-2, 3): # 5x5 Kernel
                    for dj in range(-2, 3):
                        ni, nj = i + di, j + dj
                        if (ni, nj) in raw_grid:
                            cell = raw_grid[(ni, nj)]
                            
                            # Gaussian Weight (sigma = 1.0 bin unit)
                            dist_sq = di**2 + dj**2
                            weight = math.exp(-dist_sq / 2.0)
                            
                            w_sum_ox += cell['sum_ox'] * weight 
                            w_sum_oy += cell['sum_oy'] * weight
                            w_sum_sq_ox += cell['sum_sq_ox'] * weight
                            w_sum_sq_oy += cell['sum_sq_oy'] * weight
                            w_count += cell['count'] * weight
                
                if w_count > 0:
                    mx = w_sum_ox / w_count
                    my = w_sum_oy / w_count
                    
                    # Variance = E[X^2] - (E[X])^2
                    # Note: Using population variance formula for weighted estimation
                    mean_sq_x = w_sum_sq_ox / w_count
                    mean_sq_y = w_sum_sq_oy / w_count
                    
                    var_x = max(0, mean_sq_x - mx**2)
                    var_y = max(0, mean_sq_y - my**2)
                    
                    std_x = math.sqrt(var_x)
                    std_y = math.sqrt(var_y)
                    
                    real_n = raw_grid.get((i,j), {}).get('count', 0)
                    
                    # Threshold for saving bin
                    if w_count >= 1.0:
                        
                        # NO DAMPING - We trust the empirical data + variance
                        # We do retain the bin centers for reference if needed, but we output the calculated Mean/Std
                        
                        k = f"{int(i)}_{int(j)}"
                        model[k] = {
                            'mx': float(mx),
                            'my': float(my),
                            'std_x': float(std_x),
                            'std_y': float(std_y),
                            'n': int(real_n),
                            'w_n': float(w_count)
                        }
                        valid_bins_count += 1
            
        print(f"Model {role} trained with {valid_bins_count} bins.")
        
        model_struct = {
            'meta': {
                'bin_size': BIN_SIZE,
                'x_min': 0, 'x_max': 100,
                'y_min': -50, 'y_max': 50,
                'role': role
            },
            'bins': model
        }
        
        # Save F/D specific model
        # Output: blocked_shot_model_F.json
        fname = MODEL_OUT.replace('.json', f'_{role}.json')
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, 'w') as f:
            json.dump(model_struct, f, indent=2)
            
        print(f"Saved model to {fname}")

if __name__ == "__main__":
    main()
