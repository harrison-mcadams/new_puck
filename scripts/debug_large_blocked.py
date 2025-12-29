
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from puck import fit_xgs, correction

def debug():
    print("Loading ONLY 20252026 data for debug...")
    # Load specific file to be fast
    data_path = Path('c:/Users/harri/Desktop/new_puck/data/20252026.csv')
    if not data_path.exists():
        print(f"File not found: {data_path}")
        return

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows.")
    
    # Enrich (needed for correction logic? No, correction only needs team/Coords/Side)
    # But let's check what cols we have
    req_cols = ['event', 'team_id', 'home_id', 'away_id', 'home_team_defending_side', 'x', 'y', 'distance']
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        print(f"WARNING: Missing columns: {missing}")
    
    print("Applying Correction (with new Coordinate Flip)...")
    df_fixed = correction.fix_blocked_shot_attribution(df)
    
    # Inspect
    mask_blocked = (df_fixed['event'] == 'blocked-shot')
    mask_large = (df_fixed['distance'] > 100) & mask_blocked
    
    n_large = mask_large.sum()
    print(f"\nBlocked Shots > 100ft: {n_large}")
    
    if n_large > 0:
        print("\n--- SAMPLE OF ERROR ROWS ---")
        show_cols = ['game_id', 'period', 'event', 'team_id', 'home_team_defending_side', 'x', 'y', 'distance']
        print(df_fixed.loc[mask_large, show_cols].head(20).to_string())
        
    print("\n--- Success Check (Distance < 100) ---")
    mask_good = (df_fixed['distance'] <= 100) & mask_blocked
    if mask_good.any():
        print(df_fixed.loc[mask_good, ['x', 'y', 'distance']].describe())
        print("Sample of Corrected Rows:")
        print(df_fixed.loc[mask_good, ['x', 'y', 'distance', 'home_team_defending_side']].head())

if __name__ == "__main__":
    debug()
