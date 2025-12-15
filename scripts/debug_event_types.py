
import sys
import os
import pandas as pd
import numpy as np

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import timing
from puck import analyze

def main():
    print("--- Debugging xG Sum by Event Type ---")
    
    game_id = 2025020028
    season = '20252026'
    
    # Load Game
    df_data = timing.load_season_df(season)
    df_game = df_data[df_data['game_id'] == game_id].copy()
    
    # Predict xG
    df_game, _, _ = analyze._predict_xgs(df_game)
    
    # Group by Event Type
    print("\nxG Stats by Event Type (All Game States):")
    stats = df_game.groupby('event')['xgs'].agg(['count', 'sum', 'mean']).sort_values('sum', ascending=False)
    print(stats)
    
    # Filter to 5v5 Intervals to simulate the cache condition
    condition = {'game_state': ['5v5'], 'is_net_empty': [0]}
    common_intervals = timing.get_game_intervals_cached(game_id, season, condition)
    
    # Apply intervals locally roughly (to just see the effect)
    # We can use analyze._apply_intervals if accessible, or just manual mock
    # Just inspect the global for now, if "Faceoff" has 2.0 xG total, that's the smoking gun.
    
    total_xg = df_game['xgs'].sum()
    shot_events = ['goal', 'shot-on-goal', 'missed-shot', 'blocked-shot']
    shot_xg = df_game[df_game['event'].isin(shot_events)]['xgs'].sum()
    
    print(f"\nTotal xG (All Events): {total_xg:.4f}")
    print(f"Shot xG (Shots Only): {shot_xg:.4f}")
    print(f"Non-Shot xG: {total_xg - shot_xg:.4f}")
    
    if (total_xg - shot_xg) > 1.0:
        print("\n[CONFIRMED] Non-shot events contribute significantly to total xG.")
    else:
        print("\n[UNCERTAIN] Non-shot xG is small.")

if __name__ == "__main__":
    main()
