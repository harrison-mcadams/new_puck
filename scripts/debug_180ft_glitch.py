
import pandas as pd
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from puck import config, parse

def debug_180_glitch():
    # Load raw data for Game 2025020107 (Short 20107) - UTA vs NYR
    # Lawson Crouse shot at P3 2440s (OT?) No, period 3.
    # Time was 2440 sec? 3 * 1200 = 3600. wait.
    # game_seconds in deep dive was 2440. That's P3 00:40? 
    # (3-1)*1200 + sec?
    # No, game_seconds is 0-3600.
    # 2440 / 60 = 40.6 mins. That's Period 3, 00:40.
    # Let's find the event in the loaded df.
    
    season_df_path = os.path.join(config.DATA_DIR, '20252026.csv')
    df = pd.read_csv(season_df_path)
    
    # Filter for game 2025020107
    game_df = df[df['game_id'] == 2025020107].copy()
    
    if game_df.empty:
        print("Game not found in season data.")
        return
        
    print(f"Loaded {len(game_df)} events for Game 2025020107.")
    
    # Calc game_seconds
    if 'time_elapsed_in_period_s' in game_df.columns:
        game_df['game_seconds'] = (game_df['period'] - 1) * 1200 + game_df['time_elapsed_in_period_s']
    elif 'period_seconds' in game_df.columns:
        game_df['game_seconds'] = (game_df['period'] - 1) * 1200 + game_df['period_seconds']
    else:
        print("No time columns found:", game_df.columns)
        return
    
    # Find Lawson Crouse shots
    # shooterName 'Lawson Crouse'
    # We might only have player_id. 
    # Need to find ID for Lawson Crouse or just look at all shots around 2440s.
    # game_seconds should be column
    
    target_time = 2440.0
    tolerance = 10.0
    
    print("\n--- Index Slice 30120:30140 (CSV) ---")
    sub = game_df.loc[30120:30140]
    cols_to_show = ['period', 'time_elapsed_in_period_s', 'event', 'x', 'y', 'distance', 'team_id', 'player_name']
    cols_to_show = [c for c in cols_to_show if c in game_df.columns]
    print(sub[cols_to_show].to_csv(index=True))
    
    # We also want to verify if correction logic was applied?
    # parse.py does NOT apply correction.py automatically? 
    # Usually pipelines usage: parse -> df -> clean -> predict?
    # Wait, where is `fix_blocked_shot_attribution` called?
    # It is usually called in `analyze._predict_xgs` or `process_daily_cache`.
    # BUT, the 20252026.csv we loaded comes from `run_league_stats` -> `timing.load_season_df` -> which loads the csv.
    # Does `run_league_stats` apply correction?
    # Let's check the coordinates in the CSV.
    
if __name__ == "__main__":
    debug_180_glitch()
