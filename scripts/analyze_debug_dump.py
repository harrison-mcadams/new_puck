
import numpy as np
import os
import sys

def analyze_dump():
    fpath = 'debug_chi_live.npz'
    if not os.path.exists(fpath):
        print("Dump not found.")
        return

    print(f"Loading {fpath}...")
    with np.load(fpath, allow_pickle=True) as data:
        print("Keys:", list(data.keys()))
        
        grid = data['grid']
        team_norm = data['team_norm']
        league_norm = data['league_norm']
        rel_grid = data['rel_grid']
        
        print("\n--- Stats ---")
        print(f"Grid: Min={np.min(grid)}, Max={np.max(grid)}, Mean={np.mean(grid)}, HasNan={np.isnan(grid).any()}")
        print(f"Team Norm: Min={np.min(team_norm)}, Max={np.max(team_norm)}, Mean={np.mean(team_norm)}")
        print(f"League Norm: Min={np.min(league_norm)}, Max={np.max(league_norm)}, Mean={np.mean(league_norm)}")
        print(f"Rel Grid: Min={np.min(rel_grid)}, Max={np.max(rel_grid)}, Mean={np.mean(rel_grid)}")
        
        # Check distribution of Rel Grid
        # Is it all zeros?
        print(f"Rel Grid Abs Max: {np.max(np.abs(rel_grid))}")
        
        # Check unique values
        uniq = np.unique(rel_grid)
        print(f"Unique values count: {len(uniq)}")
        if len(uniq) < 20:
            print(f"Unique values: {uniq}")

if __name__ == "__main__":
    analyze_dump()
