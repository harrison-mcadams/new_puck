
import sys
import os
import pandas as pd
import numpy as np
import json

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import timing
from puck import analyze

def verify_player(game_id, player_id, season='20252026'):
    print(f"--- Verifying Player {player_id} in Game {game_id} ---")
    
    # 1. Load Data
    df_shifts = timing._get_shifts_df(int(game_id), season=season)
    p_shifts = df_shifts[df_shifts['player_id'] == int(player_id)]
    
    if p_shifts.empty:
        print("Player has no shifts in this game.")
        return

    print(f"Found {len(p_shifts)} shifts.")
    
    # 2. Compute TOI manually from shifts
    toi_manual = (p_shifts['end_total_seconds'] - p_shifts['start_total_seconds']).sum()
    print(f"Manual TOI Sum: {toi_manual:.2f} seconds ({toi_manual/60:.2f} mins)")
    
    # 3. Intervals
    intervals_home = timing.get_game_intervals_cached(game_id, season, {'game_state': ['5v5'], 'is_net_empty': [0]})
    # Assume player is home team for simplicity of interval fetching (or just fetch both and see which one player is on)
    
    # Find team
    team_id = p_shifts.iloc[0]['team_id']
    print(f"Player Team ID: {team_id}")
    
    # Get intervals for 5v5
    # We need to know if team is Home or Away to know which '5v5' applies (Global 5v5 is safe).
    # ...
    
    # 4. Run Analysis Logic (Simulated)
    # Replicate process_daily_cache.py logic
    
    # Load Game Data
    df_game = pd.read_csv(f'data/{season}.csv') # Slow, but robust for verification
    df_game = df_game[df_game['game_id'] == int(game_id)]
    print(f"Game Events: {len(df_game)}")

    # Construct Intervals Input
    # This is tricky without replicating all flip logic.
    # Let's just use what process_daily_cache does.
    
    # ... actually, let's just inspect the CACHE file if it exists.
    cache_path = f"data/cache/{season}/partials/{game_id}_5v5.npz"
    if os.path.exists(cache_path):
        print(f"Loading Partial Cache: {cache_path}")
        with np.load(cache_path) as data:
            k_stats = f"p_{player_id}_stats"
            if k_stats in data:
                stats = json.loads(str(data[k_stats]))
                print("\n--- Cache Stats ---")
                print(json.dumps(stats, indent=2))
            else:
                print(f"Key {k_stats} not found in cache.")

            k_grid_tm = f"p_{player_id}_grid_team"
            if k_grid_tm in data:
                print(f"\nGrid Team Sum: {np.sum(data[k_grid_tm]):.4f}")
            else:
                print("\nGrid Team NOT FOUND")

            k_grid_ot = f"p_{player_id}_grid_other"
            if k_grid_ot in data:
                print(f"Grid Other Sum: {np.sum(data[k_grid_ot]):.4f}")
            else:
                print("Grid Other NOT FOUND")
    else:
        print("Cache file not found.")

if __name__ == "__main__":
    # Find a game and player dynamically
    print("Selecting a random player/game...")
    df = pd.read_csv('data/20252026.csv', nrows=10000)
    # Get a game
    gid = df['game_id'].unique()[0]
    
    # Get shifts for that game
    shifts = timing._get_shifts_df(int(gid), season='20252026')
    if not shifts.empty:
        pid = shifts['player_id'].iloc[0]
        verify_player(gid, pid)
    else:
        print("No shifts found for sample game.")
