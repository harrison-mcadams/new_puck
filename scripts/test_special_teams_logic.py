
import sys
import os
import shutil
import numpy as np
import pandas as pd
import json

# Add scripts and root to path
sys.path.append(os.path.abspath('scripts'))
sys.path.append(os.path.abspath('.'))

import process_daily_cache
from puck import timing

def test_special_teams_logic():
    season = '20252026'
    game_id = 2025020007 # PHI vs VAN
    cond_name = '5v4' # Home PP, Away PP
    
    print(f"--- Testing Special Teams Logic for {game_id} [{cond_name}] ---")
    
    # Ensure Test Dir
    partials_dir = 'data/cache/20252026/test_partials'
    os.makedirs(partials_dir, exist_ok=True)
    
    # Load Season Data to get game DF
    df_data = timing.load_season_df(season)
    df_game = df_data[df_data['game_id'] == game_id]
    
    if df_game.empty:
        print("Game not found in season data.")
        return

    # Condition
    # 5v4 = Home 5, Away 4
    # With fix, Away Team should flip to 4v5 (Home 4, Away 5)
    condition = {'game_state': ['5v4'], 'is_net_empty': [0]}
    
    # Force process
    path = process_daily_cache.get_game_partials_path(partials_dir, game_id, cond_name)
    if os.path.exists(path):
        os.remove(path)
        
    print("Running process_game...")
    success = process_daily_cache.process_game(
        game_id, df_game, season, condition, partials_dir, cond_name, force=True
    )
    
    if not success:
        print("process_game reported failure.")
        return
        
    if not os.path.exists(path):
        print("Output file not created.")
        return
        
    print(f"Output created at {path}")
    
    # Inspect Results
    with np.load(path, allow_pickle=True) as data:
        keys = list(data.keys())
        # print("Keys:", keys)
        
        # Identify Home and Away logic
        home_id = df_game.iloc[0]['home_id']
        away_id = df_game.iloc[0]['away_id']
        print(f"Home: {home_id}, Away: {away_id}")
        
        if f"team_{home_id}_stats" in data:
            s_home = json.loads(str(data[f"team_{home_id}_stats"]))
            t_home = s_home.get('team_seconds', 0)
        else:
            s_home = {}
            t_home = 0.0
            
        if f"team_{away_id}_stats" in data:
            s_away = json.loads(str(data[f"team_{away_id}_stats"]))
            t_away = s_away.get('team_seconds', 0)
        else:
            s_away = {}
            t_away = 0.0
        
        print(f"Home PP Time (5v4): {t_home:.1f}s")
        print(f"Away PP Time (4v5): {t_away:.1f}s")
        
        if t_home == t_away:
            print("[WARNING] Times are identical. Could be coincidence, or logic failed to flip.")
            if t_home == 0:
                print("[FAIL] Both zero.")
        else:
            print("[PASS] Times differ, implying asymmetric intervals were used.")
            
        if t_home > 0 and t_away > 0:
            print("[PASS] Both teams have PP time detected.")
        else:
            print("[INFO] One team had 0 PP time (Plausible but check boxscore).")
            
        print("\nTest Complete.")

if __name__ == "__main__":
    test_special_teams_logic()
