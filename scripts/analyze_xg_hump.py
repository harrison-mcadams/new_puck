
import pandas as pd
import numpy as np
import os
import sys

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from puck import config

import sys
import io

def main():
    output = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = output
    
    try:
        run_analysis()
    finally:
        sys.stdout = old_stdout
        
    res = output.getvalue()
    print(res)
    with open('hump_analysis.txt', 'w') as f:
        f.write(res)
    print("\nResults saved to hump_analysis.txt")

def run_analysis():
    csv_path = os.path.join(config.ANALYSIS_DIR, 'shot_comparison_deep_dive_2025.csv')
    if not os.path.exists(csv_path):
        print(f"Data file not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Filter for non-empty net
    if 'is_net_empty' in df.columns:
        df = df[df['is_net_empty'] == 0]
        
    print(f"Analyzing {len(df)} non-empty net shots...")
    
    # Define bins
    bins = [0.0, 0.05, 0.10, 0.15, 1.0]
    labels = ['0.0-0.05', '0.05-0.10 (Hump)', '0.10-0.15', '0.15+']
    df['xg_bin'] = pd.cut(df['xgs'], bins=bins, labels=labels)
    
    # Count shots in each bin
    print("\nShot Counts by xG Bin:")
    print(df['xg_bin'].value_counts().sort_index().to_string())
    
    # Deep dive into the "Hump" bin (0.05-0.10)
    hump = df[df['xg_bin'] == '0.05-0.10 (Hump)']
    pre_hump = df[df['xg_bin'] == '0.0-0.05']
    
    print(f"\n--- Composition of the 'Hump' (0.05-0.10) vs Low Danger (0.0-0.05) ---")
    
    # 1. Shot Types
    print("\nTop Shot Types (Hump):")
    print(hump['shot_type'].value_counts(normalize=True).head(5).to_string())
    print("\nTop Shot Types (Low Danger - Pre-Hump):")
    print(pre_hump['shot_type'].value_counts(normalize=True).head(5).to_string())
    
    # 2. Distance Distribution
    print(f"\nMean Distance: Hump={hump['distance'].mean():.1f} ft, Pre-Hump={pre_hump['distance'].mean():.1f} ft")
    print(f"Median Distance: Hump={hump['distance'].median():.1f} ft, Pre-Hump={pre_hump['distance'].median():.1f} ft")
    
    # 3. Angle Distribution
    print(f"Mean Angle: Hump={hump['angle_deg'].abs().mean():.1f}, Pre-Hump={pre_hump['angle_deg'].abs().mean():.1f}")
    
    # 4. Secondary Features (Rebounds, Rush)
    if 'is_rebound' in hump.columns:
        print(f"\nRebound %: Hump={hump['is_rebound'].mean():.1%}, Pre-Hump={pre_hump['is_rebound'].mean():.1%}")
    if 'is_rush' in hump.columns:
        print(f"Rush %: Hump={hump['is_rush'].mean():.1%}, Pre-Hump={pre_hump['is_rush'].mean():.1%}")

    # Check for specific dominant clusters
    # e.g. Wrist shots from 20-40ft?
    wrist_hump = hump[hump['shot_type'] == 'wrist']
    if not wrist_hump.empty:
        print(f"\nWrist Shots in Hump: Mean Dist={wrist_hump['distance'].mean():.1f}, Count={len(wrist_hump)}")
        
    snap_hump = hump[hump['shot_type'] == 'snap']
    if not snap_hump.empty:
        print(f"Snap Shots in Hump: Mean Dist={snap_hump['distance'].mean():.1f}, Count={len(snap_hump)}")

if __name__ == "__main__":
    main()
