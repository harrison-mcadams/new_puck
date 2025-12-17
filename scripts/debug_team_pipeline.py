import sys
import os
import argparse
import pandas as pd
import numpy as np

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import timing
from puck import parse
from puck import config
from scripts import process_daily_cache
from scripts import run_league_stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--team', type=str, default='PHI', help='Team abbreviation')
    parser.add_argument('--season', type=str, default='20252026')
    args = parser.parse_args()
    
    team = args.team
    season = args.season
    
    print(f"--- Debugging Pipeline for {team} ({season}) ---")
    
    '''
    # 1. Load/Fetch Data (Assume fetch was partially done or just fetch needed games? 
    # For simplicity, we load the season CSV if exists, or fetch whole thing if needed. 
    # Fetching whole thing is usually fast if cached, but let's try to trust existing data first)
    
    print("[1] Loading Season Data...")
    df_season = parse._season(season=season, out_path='data', use_cache=True)
    
    if df_season.empty:
        print("No data found!")
        return
        
    # Filter for games involving team
    # We need home_abb or away_abb. 
    # parse._season returns game_id, team_id, etc.
    # We usually rely on a team_id mapping, but let's filter by abbr if available.
    
    # Actually, let's process ALL rows for games involving the team.
    # We need to find the game_ids first.
    
    # df_season has 'home_abb', 'away_abb' usually if enriched?
    # No, _season returns play-by-play.
    # But it usually has 'home_team_id' or similar?
    # Let's use the game list from timing if possible or just unique games.
    
    all_game_ids = df_season['game_id'].unique()
    print(f"Total games in data: {len(all_game_ids)}")
    
    # We need to identify which games involve PHI.
    # df_season doesn't strictly have abbreviations in every row maybe?
    # But it should have 'home_name' / 'away_name' or similar.
    # Let's lazily process ALL games for the cache, but ONLY for that team? 
    # No, that's slow.
    
    # Optimization: Filter game_ids where team is playing.
    # We can check the first event of each game to find teams.
    print("[2] Identifying Tables...")
    target_games = []
    
    # To save time, let's group by game_id and check.
    # Or iterate unique games.
    
    # Let's enable xG prediction on the WHOLE df first to be safe (vectorized is fast)
    print("[2a] Predicting xG (Robust)...")
    from puck import analyze
    # Force overwrite behavior to ensure our FIX is applied
    df_season, _, _ = analyze._predict_xgs(df_season, behavior='overwrite')
    
    print("[3] Filtering Games for Team...")
    # Get team ID if possible, or matches string
    # Let's search for rows where home_abb or away_abb == TEAM
    # df_season cols: ...
    mask_team = (df_season['home_abb'] == team) | (df_season['away_abb'] == team)
    target_games = df_season.loc[mask_team, 'game_id'].unique()
    
    print(f"Found {len(target_games)} games for {team}")
    
    if len(target_games) == 0:
        print(f"No games found for {team}. Check abbreviation.")
        # Try to show unique abbs
        print("Available teams:", df_season['home_abb'].dropna().unique())
        return

    # 4. Run Cache Process ONLY for these games
    print(f"[4] Processing Daily Cache for {len(target_games)} games...")
    partials_dir = process_daily_cache.ensure_dirs(season)
    
    # We need to run for 5v5 (and others if desired, but 5v5 is main)
    cond_name = '5v5'
    condition = {'game_state': ['5v5'], 'is_net_empty': [0]}
    
    # Ensure intervals cached (fast loop)
    # print("  -> Pre-computing intervals...")
    # for gid in target_games:
    #    timing.get_game_intervals_cached(gid, season, condition)
        
    print("  -> Processing .npz caches...")
    count = 0
    for gid in target_games:
        df_game = df_season[df_season['game_id'] == gid]
        # Force processing
        process_daily_cache.process_game(gid, df_game, season, condition, partials_dir, cond_name, force=True)
        count += 1
        if count % 10 == 0:
            print(f"     Processed {count}/{len(target_games)}")
            
    '''
            
    print("[5] Running League Stats (Aggregation)...")
    # run_league_stats takes a --teams arg
    # calling verify logic essentially
    
    # We can call the main function or subprocess
    # subprocess is easier to ensuring args parsing
    cmd = [
        sys.executable, 
        'scripts/run_league_stats.py', 
        '--season', season
    ]
    import subprocess
    subprocess.run(cmd, check=True)
    
    print(f"Done! Check analysis/xleague/{season}/{cond_name}/{team}_relative_map.png")

if __name__ == "__main__":
    main()
