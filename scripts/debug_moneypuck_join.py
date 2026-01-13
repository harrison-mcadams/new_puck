import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))
from puck import moneypuck, fit_xgs

print("--- Debugging MoneyPuck Join ---")

# 1. Load Local Data (Small Sample)
print("Loading local data sample...")
try:
    # Try loading from the processed test file from training? 
    # Or just load raw
    # Let's load a specific season to be sure
    season = "20252026"
    df_local = fit_xgs.load_all_seasons_data(base_dir='data') 
    # Filter for season if possible, or just sample
    if 'game_id' in df_local.columns:
         df_local = df_local[df_local['game_id'].astype(str).str.startswith('2025')]
    
    df_local = df_local.sample(100, random_state=42)
    print(f"Local Sample: {len(df_local)}")
    print("Local Columns:", df_local.columns.tolist())
    
    print("\nLocal GameID Examples:")
    print(df_local['game_id'].head())
    
    # Calculate game_seconds if needed
    if 'total_time_elapsed_s' in df_local.columns:
        print("\nLocal Time (total_time_elapsed_s) Examples:")
        print(df_local['total_time_elapsed_s'].head())
    
except Exception as e:
    print(f"Failed to load local: {e}")
    sys.exit(1)

# 2. Download MP Data
print("\nDownloading MoneyPuck 2025...")
try:
    df_mp = moneypuck.download_shots("2025")
    print(f"MP Size: {len(df_mp)}")
    print("MP Columns:", df_mp.columns.tolist())
    
    print("\nMP GameID Examples:")
    print(df_mp['game_id'].head())
    
    print("\nMP Time Examples:")
    print(df_mp['time'].head())

except Exception as e:
    print(f"Failed to load MP: {e}")
    sys.exit(1)

# 3. Attempt Merge
print("\nAttempting Merge...")
try:
    merged = moneypuck.merge_predictions(df_local, df_mp)
    print(f"Merged Result: {len(merged)}")
    print(f"Matched: {merged['mp_shotID'].notna().sum()}")
    
    if merged['mp_shotID'].notna().sum() == 0:
        print("\nDEBUG: Join Keys Mismatch Analysis")
        # Check IDs
        loc_ids = df_local['game_id'].astype(int).unique()
        mp_ids = df_mp['game_id'].astype(int).unique()
        
        print(f"Local ID Format: {loc_ids[0]}")
        print(f"MP ID Format:    {mp_ids[0]}")
        
        overlap = np.intersect1d(loc_ids, mp_ids)
        print(f"Overlapping GameIDs: {len(overlap)}")
        
except Exception as e:
    print(f"Merge failed: {e}")
    import traceback
    traceback.print_exc()
