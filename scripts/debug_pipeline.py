
import sys
import os
import pandas as pd
import numpy as np

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import timing
from puck import analyze

def main():
    print("--- Debugging Pipeline Leakage ---")
    
    # Pick a game known to have issues?
    # Use one from the file list: 2025020028 (ANA case from before? No that was summary)
    # 2025020028_5v5.npz had 9.8 xG/60.
    game_id = 2025020028
    season = '20252026'
    condition_name = '5v5'
    
    # 1. Load Game Data
    print(f"Loading Game {game_id}...")
    df_data = timing.load_season_df(season)
    df_game = df_data[df_data['game_id'] == game_id].copy()
    
    # Predict xG (simulate daily.py pre-load)
    print("Predicting xG...")
    df_game, _, _ = analyze._predict_xgs(df_game)
    
    # 2. Get Intervals
    print("Fetching Intervals...")
    conditions_map = {
        '5v5': {'game_state': ['5v5'], 'is_net_empty': [0]}
    }
    condition = conditions_map[condition_name]
    common_intervals = timing.get_game_intervals_cached(game_id, season, condition)
    
    # Calculate TOI
    toi = sum(e-s for s,e in common_intervals)
    print(f"Common Intervals TOI: {toi:.1f}s")
    
    # 3. Simulate process_daily_cache.py (Current - LEAKY?)
    print("\n--- Current Logic (Time Intervals Only) ---")
    intervals_input = {
        'per_game': {
            str(game_id): {
                'intersection_intervals': common_intervals
            }
        }
    }
    
    # Team ID from game (First row)
    home_id = df_game.iloc[0]['home_id']
    print(f"Home Team ID: {home_id}")
    
    # Condition passed is just team
    analysis_condition_current = {'team': home_id}
    
    # Run xgs_map
    _, _, df_filt_current, stats_current = analyze.xgs_map(
        season=season,
        data_df=df_game,
        intervals_input=intervals_input,
        condition=analysis_condition_current,
        heatmap_only=True
    )
    
    xg_current = stats_current['team_xgs']
    print(f"Current Team xG: {xg_current:.4f}")
    print(f"Current xG/60: {(xg_current / toi * 3600):.2f}")
    
    # 4. Simulate Proposed Fix (Explicit Game State Filter)
    print("\n--- Proposed Logic (Values + Game State) ---")
    analysis_condition_fixed = {'team': home_id}
    # Merge in global condition logic
    analysis_condition_fixed.update(condition) 
    # {'team': HID, 'game_state': ['5v5'], 'is_net_empty': [0]}
    
    _, _, df_filt_fixed, stats_fixed = analyze.xgs_map(
        season=season,
        data_df=df_game,
        intervals_input=intervals_input,
        condition=analysis_condition_fixed,
        heatmap_only=True
    )
    
    xg_fixed = stats_fixed['team_xgs']
    print(f"Fixed Team xG: {xg_fixed:.4f}")
    print(f"Fixed xG/60: {(xg_fixed / toi * 3600):.2f}")
    
    # 5. Difference Analysis
    diff = xg_current - xg_fixed
    if diff > 0.01:
        print(f"\n[LEAK DETECTED] Difference: {diff:.4f}")
        print("Events causing leak:")
        # Find rows in current not in fixed
        leak_indices = set(df_filt_current.index) - set(df_filt_fixed.index)
        leaked_rows = df_filt_current.loc[list(leak_indices)]
        print(leaked_rows[['event', 'period', 'game_state', 'xgs']])
    else:
        print("\n[NO LEAK] Logic seems equivalent.")

if __name__ == "__main__":
    main()
