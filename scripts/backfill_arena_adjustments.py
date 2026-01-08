
import os
import sys
import pandas as pd
import glob

# Add project root to path to import puck
sys.path.append(os.getcwd())
from puck import arena_adjustments
from puck import nhl_api

def get_home_team_map(season):
    """
    Helper to create a GameID -> HomeTeamName map for a season.
    We need this because the CSVs might only have 'home_abb' (abbreviation),
    but the adjuster needs the full Name (e.g. 'Rangers').
    
    We can fetch the schedule once per season.
    """
    print(f"  Fetching schedule for {season} to map GameID -> HomeTeamName...")
    games = nhl_api.get_season(season=season, team='all') # Returns list of game dicts
    game_map = {}
    for g in games:
        gid = g.get('gamePk') or g.get('id')
        # Extract home team name
        # 1. Try New API (homeTeam.commonName.default) -> returns Nickname (e.g. "Lightning")
        #    This matches the JSON keys usually.
        name = None
        try:
            name = g.get('homeTeam', {}).get('commonName', {}).get('default')
        except:
            pass
            
        # 2. Try simple name (if available)
        if not name:
             name = g.get('homeTeam', {}).get('name')

        # 3. Try Legacy structure
        if not name:
            h = g.get('teams', {}).get('home', {}).get('team', {})
            name = h.get('name')
            
        if gid and name:
            game_map[gid] = name
            
    return game_map

def process_season(season_dir):
    season_name = os.path.basename(season_dir)
    csv_path = os.path.join(season_dir, f"{season_name}_df.csv")
    
    if not os.path.exists(csv_path):
        print(f"Skipping {season_name} (No CSV found)")
        return
        
    print(f"Processing {season_name}...")
    
    # 1. Load CSV
    df = pd.read_csv(csv_path)
    if 'x' not in df.columns or 'y' not in df.columns:
        print(f"  Skipping {season_name} (No coords)")
        return

    # 2. Get Home Team Map
    # The CSV has 'home_abb', but 'arena_adjustments.json' uses 'Rangers', 'Maple Leafs' etc.
    # We shouldn't rely on abbreviations because they change (PHX->ARI->UTA).
    # We need the Name.
    game_home_map = get_home_team_map(season_name)
    
    # 3. Apply Adjustments
    # Vectorized approach is hard because adjustment depends on 'arena' which depends on 'game_id'.
    # Iteration is safer.
    
    x_adjs = []
    y_adjs = []
    
    # Cache for game_id to minimize lookups
    current_game_id = None
    current_home_team = None
    
    shot_types = ['shot-on-goal', 'missed-shot', 'goal']
    
    count_adj = 0
    
    for idx, row in df.iterrows():
        gid = row.get('game_id')
        evt = row.get('event')
        
        # Optimize: Only check game map if game_id changes
        if gid != current_game_id:
            current_game_id = gid
            current_home_team = game_home_map.get(gid, 'Unknown')
            
        x, y = row.get('x'), row.get('y')
        
        # Logic: Only adjust shots
        if evt in shot_types:
            try:
                xa, ya = arena_adjustments.adjust_shot(x, y, current_home_team, season_name)
                if xa != x or ya != y:
                    count_adj += 1
                
                # Debug first few attempts per season
                if idx < 50 and count_adj < 5:
                     print(f"    [DEBUG] Game {gid} | Home: '{current_home_team}' | ({x},{y}) -> ({xa},{ya}) | Diff: {xa!=x}")
            except Exception as e:
                if idx < 5:
                    print(f"    [ERROR] Adjustment failed: {e}")
                xa, ya = x, y
        else:
            xa, ya = x, y
            
        x_adjs.append(xa)
        y_adjs.append(ya)
        
    df['x_adj'] = x_adjs
    df['y_adj'] = y_adjs
    
    # 4. Save
    df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}. Adjusted {count_adj} events.")

def main():
    data_dir = "data"
    seasons = glob.glob(os.path.join(data_dir, "20*"))
    
    for s_dir in seasons:
        if os.path.isdir(s_dir):
            process_season(s_dir)

if __name__ == "__main__":
    main()
