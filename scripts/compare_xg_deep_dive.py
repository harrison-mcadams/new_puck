"""compare_xg_deep_dive.py

Deep dive analysis of xG differences between our NestedGLM (Spline) and MoneyPuck.
Focuses on unblocked shots to isolate model logic differences.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

def main():
    print("--- xG Comparison Deep Dive ---")
    
    # 1. Load Data
    pred_path = Path('analysis/nested_xgs/test_predictions.csv')
    if not pred_path.exists():
        print(f"Error: {pred_path} not found. Run training first.")
        return
        
    df = pd.read_csv(pred_path)
    print(f"Loaded {len(df)} predictions.")
    
    # 2. Filter: Unblocked ONLY + Valid MoneyPuck Match
    df_clean = df[
        (df['event'] != 'blocked-shot') & 
        (df['mp_xGoal'].notna())
    ].copy()
    
    print(f"Unblocked & Matched: {len(df_clean)}")
    
    if len(df_clean) == 0:
        print("No valid data to analyze.")
        return

    # 3. Calculate Differences
    # Residual = Us - Them
    df_clean['xg_diff'] = df_clean['xG'] - df_clean['mp_xGoal']
    df_clean['abs_diff'] = df_clean['xg_diff'].abs()
    
    # 4. Input Discrepancies (Distance)
    # Check if we are even talking about the same shot location
    if 'mp_shotDistance' in df_clean.columns:
        df_clean['dist_diff'] = df_clean['distance'] - df_clean['mp_shotDistance']
        print("\n--- Input Data Agreement ---")
        print(f"Distance Corr: {df_clean['distance'].corr(df_clean['mp_shotDistance']):.4f}")
        print(f"Mean Dist Diff (Us - MP): {df_clean['dist_diff'].mean():.2f} ft")
        print(f"MAE Dist Diff: {df_clean['dist_diff'].abs().mean():.2f} ft")
        
        # Filter for "Clean Data" (Location agrees within 5ft)
        df_loc_match = df_clean[df_clean['dist_diff'].abs() < 5.0].copy()
        print(f"\nSubset with Matching Location (<5ft diff): {len(df_loc_match)} ({len(df_loc_match)/len(df_clean)*100:.1f}%)")
        print(f"Correlation (Loc Match): {df_loc_match['xG'].corr(df_loc_match['mp_xGoal']):.4f}")
    else:
        df_loc_match = df_clean
        
    # 5. Residual Analysis (on Location Matched Data)
    print("\n--- Residual Analysis (Location Matched) ---")
    
    # Binned Distance Analysis
    df_loc_match['dist_bin'] = pd.cut(df_loc_match['distance'], bins=[0, 10, 20, 30, 40, 50, 60, 100])
    bin_stats = df_loc_match.groupby('dist_bin', observed=True)[['xG', 'mp_xGoal', 'xg_diff']].mean()
    print("\nMean xG by Distance Bin:")
    print(bin_stats)
    
    # Shot Type Analysis
    if 'shot_type' in df_loc_match.columns:
        print("\nMean xG by Shot Type:")
        st_stats = df_loc_match.groupby('shot_type')[['xG', 'mp_xGoal', 'xg_diff', 'abs_diff']].agg(['mean', 'count'])
        # Filter for significant types
        st_stats = st_stats[st_stats[('xG', 'count')] > 100]
        print(st_stats)
        
    # Rebound Analysis
    if 'is_rebound' in df_loc_match.columns:
        print("\nRebound stats:")
        print(df_loc_match.groupby('is_rebound')[['xG', 'mp_xGoal', 'xg_diff']].mean())
        
    # 6. Plots
    out_dir = Path('analysis/xg_comparison')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Scatter
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_loc_match, x='mp_xGoal', y='xG', alpha=0.1, s=10)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title(f"Unblocked xG Comparison (Loc Matched)\nCorr: {df_loc_match['xG'].corr(df_loc_match['mp_xGoal']):.3f}")
    plt.savefig(out_dir / 'xg_scatter_unblocked.png')
    print(f"Saved scatter to {out_dir / 'xg_scatter_unblocked.png'}")
    
    # Residual vs Distance
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_loc_match, x='distance', y='xg_diff', label='Mean Diff (Us - MP)')
    plt.axhline(0, color='k', linestyle='--')
    plt.title("xG Difference vs Distance (positive = we are higher)")
    plt.savefig(out_dir / 'xg_diff_vs_distance.png')
    print(f"Saved diff plot to {out_dir / 'xg_diff_vs_distance.png'}")
    
    # 7. Identify Biggest Disagreements
    print("\n--- Biggest Model Disagreements (Loc Matched) ---")
    # Where we are HIGH, they are LOW
    high_us = df_loc_match.sort_values('xg_diff', ascending=False).head(10)
    print("\nWe predict HIGH, MP predicts LOW:")
    cols = ['xG', 'mp_xGoal', 'xg_diff', 'distance', 'angle_deg', 'shot_type', 'is_rebound', 'is_rush']
    # Filter cols to those that exist
    cols = [c for c in cols if c in df_loc_match.columns]
    print(high_us[cols])
    
    # Where we are LOW, they are HIGH
    high_mp = df_loc_match.sort_values('xg_diff', ascending=True).head(10)
    print("\nWe predict LOW, MP predicts HIGH:")
    print(high_mp[cols])

if __name__ == "__main__":
    main()
