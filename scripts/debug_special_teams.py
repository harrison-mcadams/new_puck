
import sys
import os
import pandas as pd
import numpy as np

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import analyze
from puck import timing

def debug_special_teams():
    season = '20252026'
    print(f"Loading data for {season}...")
    df = timing.load_season_df(season)
    
    if df is None or df.empty:
        print("No data found.")
        return

    print(f"Data shape: {df.shape}")
    
    # Run prediction (overwrite to ensure fresh model usage)
    print("Running prediction...")
    df, clf, meta = analyze._predict_xgs(df, behavior='overwrite')
    
    # Ensure is_goal exists
    if 'is_goal' not in df.columns:
         if 'event' in df.columns:
             df['is_goal'] = (df['event'] == 'goal').astype(int)
         else:
             df['is_goal'] = 0

    valid_states = ['5v5', '5v4', '4v5']
    
    print("\n--- Calibration by Game State ---")
    
    # Filter for standard game states only for cleaner view, or just group all
    stats = df.groupby('game_state').agg({
        'xgs': 'sum',
        'is_goal': 'sum',
        'event': 'count'
    }).rename(columns={'event': 'n_events'})
    
    # Filter to main ones
    stats = stats.loc[stats.index.isin(valid_states)]
    
    # Metrics
    stats['ratio'] = stats['xgs'] / stats['is_goal']
    stats['diff'] = stats['xgs'] - stats['is_goal']
    stats['xg_per_event'] = stats['xgs'] / stats['n_events']
    
    print(stats)
    
    print("\n--- Blocked Shot Context ---")
    # Check blocked shot xG contribution by state
    mask_blocked = df['event'] == 'blocked-shot'
    df_blocked = df[mask_blocked]
    
    blk_stats = df_blocked.groupby('game_state')['xgs'].sum().rename('blocked_xg')
    
    # Merge for clarity
    merged = stats.join(blk_stats)
    merged['adjusted_xg'] = merged['xgs'] - merged['blocked_xg'].fillna(0)
    merged['adjusted_ratio'] = merged['adjusted_xg'] / merged['is_goal']
    
    print(merged[['xgs', 'blocked_xg', 'adjusted_xg', 'is_goal', 'ratio', 'adjusted_ratio']])

    print("\n--- Grand Total (All States) ---")
    total_xg = df['xgs'].sum()
    total_goals = df['is_goal'].sum()
    print(f"Total xG: {total_xg:.2f}")
    print(f"Total Goals: {total_goals}")
    print(f"Ratio: {total_xg / total_goals:.4f}")

if __name__ == "__main__":
    debug_special_teams()
