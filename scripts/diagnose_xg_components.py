import sys
import os
import pandas as pd
import numpy as np

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import timing
from puck import analyze
from puck import parse

def main():
    game_id = 2025010075 # Game used in previous checks
    
    print(f"--- Diagnosing xG Components for Game {game_id} ---")
    
    # 1. Load Game Data (Raw)
    # We use parse._season to get raw data for this game if possible, or filter from season
    # Let's load season 20252026 and filter.
    print("Loading data...")
    df_season = parse._season(season='20252026', out_path='data', use_cache=True)
    df_game = df_season[df_season['game_id'] == game_id].copy()
    
    print(f"Loaded {len(df_game)} events.")
    
    # 2. Run Prediction
    print("Running _predict_xgs...")
    # Force overwrite
    df_pred, _, _ = analyze._predict_xgs(df_game, behavior='overwrite')
    
    # 3. Analysis
    print("\n--- Breakdown by Event Type ---")
    if 'xgs' not in df_pred.columns:
        print("ERROR: 'xgs' column not found!")
        return
        
    stats = df_pred.groupby('event').agg(
        count=('xgs', 'count'),
        total_xg=('xgs', 'sum'),
        avg_xg=('xgs', 'mean'),
        max_xg=('xgs', 'max')
    ).sort_values('total_xg', ascending=False)
    
    print(stats)
    
    print(f"\nTotal Game xG: {df_pred['xgs'].sum():.4f}")
    
    # 4. Top 10 xG Events
    print("\n--- Top 10 High xG Events ---")
    cols = ['event', 'period', 'period_time', 'xgs', 'x', 'y', 'distance', 'angle_deg', 'shot_type', 'is_goal']
    # Check which cols exist
    show_cols = [c for c in cols if c in df_pred.columns]
    
    top_10 = df_pred.sort_values('xgs', ascending=False).head(10)[show_cols]
    print(top_10)
    
    # 5. Check Blocked Shot Imputation specifically
    print("\n--- Blocked Shot Analysis ---")
    blocked = df_pred[df_pred['event'] == 'blocked-shot']
    if not blocked.empty:
        print(blocked[show_cols].describe())
        print("\nSample Blocked Shots:")
        print(blocked.head(5)[show_cols])
    else:
        print("No blocked shots found.")
        
    # 6. Check Non-Shot Events
    print("\n--- Non-Shot Event Check ---")
    non_shots = df_pred[~df_pred['event'].isin(['goal', 'shot-on-goal', 'missed-shot', 'blocked-shot'])]
    if not non_shots.empty:
        non_shot_xg = non_shots['xgs'].sum()
        print(f"Total xG from non-shots: {non_shot_xg:.6f}")
        if non_shot_xg > 0.001:
            print("WARNING: Non-zero xG found in non-shot events!")
            print(non_shots[non_shots['xgs'] > 0][show_cols].head())
    else:
        print("No non-shot events found.")

if __name__ == "__main__":
    main()
