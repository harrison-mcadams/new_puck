import sys
import os
import pandas as pd
import numpy as np
import logging

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import analyze
from puck import fit_xgs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def print_calibration(name, df_subset):
    total_goals = (df_subset['event'] == 'goal').sum()
    total_xg = df_subset['xgs'].sum()
    ratio = total_xg / total_goals if total_goals > 0 else 0
    
    print("-" * 60)
    print(f"CALIBRATION: {name}")
    print("-" * 60)
    print(f"Events:       {len(df_subset)}")
    print(f"Total Goals:  {total_goals}")
    print(f"Total xG:     {total_xg:.2f}")
    print(f"Difference:   {total_xg - total_goals:.2f}")
    print(f"Ratio:        {ratio:.4f}")
    
    # Blocked Shot Check
    blocked = df_subset[df_subset['event'] == 'blocked-shot']
    blocked_xg = blocked['xgs'].sum()
    print(f"Blocked xG:   {blocked_xg:.2f} ({blocked_xg/total_xg*100:.1f}% of Total xG)")
    print("-" * 60)
    print("")

def main():
    season_target = 20252026
    print("Loading ALL historical data + current season...")
    
    # 1. Load All Data Manually (to ensure 'season' column exists)
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    dfs = []
    
    # Iterate over 20XX20YY directories
    for d in sorted(os.listdir(data_dir)):
        if d.isdigit() and len(d) == 8:
            # Check standard path
            season_path = os.path.join(data_dir, d, f"{d}_df.csv")
            if not os.path.exists(season_path):
                # Check root data dir fallback
                season_path = os.path.join(data_dir, f"{d}.csv")
            
            if os.path.exists(season_path):
                print(f"Loading {d} from {season_path}...")

                try:
                    df_s = pd.read_csv(season_path)
                    df_s['season'] = int(d) # Add season column
                    dfs.append(df_s)
                except Exception as e:
                    print(f"Skipping {d}: {e}")
                    
    if not dfs:
        print("No data found.")
        return
        
    df_all = pd.concat(dfs, ignore_index=True)


    print(f"Loaded {len(df_all)} total rows. Predicting xG via analyze._predict_xgs...")
    
    # 2. Run Inference Pipeline
    # This uses the EXACT logic used in daily.py (impute -> clean -> predict)
    # We pass behavior='overwrite' to ensure fresh calculation
    try:
        df_pred, _, _ = analyze._predict_xgs(df_all, behavior='overwrite')
    except Exception as e:
        print(f"Prediction failed: {e}")
        return

    # Filter to valid events only for summary
    valid = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal']
    df_valid = df_pred[df_pred['event'].isin(valid)].copy()
    
    # 3. Splits
    df_2025 = df_valid[df_valid['season'] == season_target]
    df_history = df_valid[df_valid['season'] != season_target]
    
    # 4. Report
    print_calibration("ALL DATA (Combined)", df_valid)
    print_calibration("HISTORY (Excluding 2025-2026)", df_history)
    print_calibration("CURRENT SEASON (2025-2026)", df_2025)

if __name__ == "__main__":
    main()
