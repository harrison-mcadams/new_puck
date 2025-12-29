
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from puck import config

import sys
import io

def main():
    output = io.StringIO()
    # Redirect print to our string buffer
    old_stdout = sys.stdout
    sys.stdout = output
    
    try:
        run_analysis()
    finally:
        sys.stdout = old_stdout
        
    res = output.getvalue()
    print(res)
    with open('xg_factor_analysis.txt', 'w') as f:
        f.write(res)
    print("\nResults saved to xg_factor_analysis.txt")

def run_analysis():
    csv_path = os.path.join(config.ANALYSIS_DIR, 'shot_comparison_deep_dive_2025.csv')
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Filter for Location Matches only (to isolate other factors)
    # distance check
    df_match = df[df['dist_diff'].abs() < 2.0].copy()
    print(f"Analyzing {len(df_match)} location-matched shots (dist_diff < 2ft)...")
    
    df_match['xg_diff'] = df_match['xgs'] - df_match['xGoal']
    df_match['abs_xg_diff'] = df_match['xg_diff'].abs()
    
    print(f"\nOverall Stats for Location Matches:")
    print(f"  MAE xG: {df_match['abs_xg_diff'].mean():.4f}")
    print(f"  Median xG Diff: {df_match['xg_diff'].median():.4f}")
    print(f"  My xG Mean: {df_match['xgs'].mean():.4f}")
    print(f"  MP xG Mean: {df_match['xGoal'].mean():.4f}")
    
    print("\nMatched Columns:", df_match.columns.tolist())
    
    # factor 1: Shot Types
    # mp: shotType, local: shot_type
    df_match['shot_type_match'] = (df_match['shotType'].str.lower() == df_match['shot_type'].str.lower())
    print(f"\nShot Type Agreement: {df_match['shot_type_match'].mean():.1%}")
    
    print("\nShot Type Crosstab (Top 10 Disagreements):")
    disagreements = df_match[~df_match['shot_type_match']]
    ct = pd.crosstab(disagreements['shot_type'], disagreements['shotType'])
    print(ct.to_string())

    print("\nMean xG by My Shot Type:")
    print(df_match.groupby('shot_type')[['xgs', 'xGoal']].mean().to_string())

    print("\nMean xG by MoneyPuck Shot Type:")
    print(df_match.groupby('shotType')[['xgs', 'xGoal']].mean().to_string())
    
    # factor 2: Rebounds
    # mp: shotRebound, local: is_rebound
    if 'is_rebound' in df_match.columns and 'shotRebound' in df_match.columns:
        df_match['rebound_match'] = (df_match['is_rebound'].fillna(0).astype(int) == df_match['shotRebound'].fillna(0).astype(int))
        print(f"\nRebound Agreement: {df_match['rebound_match'].mean():.1%}")
        
        rebound_summary = df_match.groupby(['is_rebound', 'shotRebound'], observed=True).agg({
            'abs_xg_diff': ['mean', 'count'],
            'xg_diff': 'mean'
        })
        print("\nRebound Disagreement Impact:")
        print(rebound_summary.to_string())

    # factor 3: Rushes
    # mp: shotRush, local: is_rush
    if 'is_rush' in df_match.columns and 'shotRush' in df_match.columns:
        df_match['rush_match'] = (df_match['is_rush'].fillna(0).astype(int) == df_match['shotRush'].fillna(0).astype(int))
        print(f"\nRush Agreement: {df_match['rush_match'].mean():.1%}")
        
        rush_summary = df_match.groupby(['is_rush', 'shotRush'], observed=True).agg({
            'abs_xg_diff': ['mean', 'count'],
            'xg_diff': 'mean'
        })
        print("\nRush Disagreement Impact:")
        print(rush_summary.to_string())

    # factor 4: Game State (if available)
    if 'game_state' in df_match.columns:
        print("\nxG Stats by Game State:")
        gs_summary = df_match.groupby('game_state', observed=True).agg({
            'xg_diff': 'mean',
            'abs_xg_diff': 'mean',
            'xgs': 'mean',
            'xGoal': 'mean',
            'shotID': 'count'
        }).rename(columns={'shotID': 'count'})
        print(gs_summary.to_string())

    # Systematic Disagreement (Models)
    # Where location, rebound, rush all match, but xG differs
    filters = (df_match['dist_diff'].abs() < 1.0) & (df_match['shot_type_match'])
    if 'rebound_match' in df_match.columns: filters &= df_match['rebound_match']
    if 'rush_match' in df_match.columns: filters &= df_match['rush_match']
    
    df_pure = df_match[filters].copy()
    print(f"\nAnalyzing {len(df_pure)} 'Purely Matched' shots (Location, Type, State all match)...")
    print(f"  Pure MAE xG: {df_pure['abs_xg_diff'].mean():.4f}")
    
    # By distance bins
    df_pure['dist_bin'] = pd.cut(df_pure['distance'], bins=[0, 10, 20, 30, 50, 100])
    print("\nxG Discrepancy by Distance Bin (Pure Matches):")
    dist_summary = df_pure.groupby('dist_bin').agg({
        'xg_diff': 'mean',
        'abs_xg_diff': 'mean',
        'xgs': 'mean',
        'xGoal': 'mean',
        'shotID': 'count'
    }).rename(columns={'shotID': 'count'})
    print(dist_summary.to_string())

    # Save examples of pure disagreement
    out_file = os.path.join(config.ANALYSIS_DIR, 'pure_xg_disagreements.csv')
    df_pure[df_pure['abs_xg_diff'] > 0.05].sort_values('abs_xg_diff', ascending=False).to_csv(out_file, index=False)
    print(f"\nSaved pure disagreements to {out_file}")

if __name__ == "__main__":
    main()
