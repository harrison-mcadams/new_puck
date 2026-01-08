
import os
import sys
import pandas as pd
import json

# Add project root to path
sys.path.append(os.getcwd())
from puck import arena_adjustments
from puck import nhl_api

def debug_season(season):
    print(f"Debugging Season: {season}")
    
    # 1. Fetch Schedule & Map
    print("Fetching schedule...")
    games = nhl_api.get_season(season=season)
    game_map = {}
    print(f"  Found {len(games)} games from API.")
    if len(games) > 0:
        sample = games[0]
        # Debug structure
        h = sample.get('teams', {}).get('home', {}).get('team', {})
        print(f"  Sample Game 0 Home Team Struct: {h}")
        print(f"  Sample Game 0 Home Team Name: {h.get('name')}")
    
    for g in games:
        gid = g.get('gamePk') or g.get('id')
        h = g.get('teams', {}).get('home', {}).get('team', {})
        name = h.get('name')
        if not name:
             name = g.get('homeTeam', {}).get('name')
             
        if gid and name:
            game_map[gid] = name
            
    print(f"  Mapped {len(game_map)} games.")
    
    # 2. Load CSV
    csv_path = os.path.join("data", season, f"{season}_df.csv")
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"  Loaded CSV: {len(df)} rows.")
    
    # 3. Test Adjustment on first few shots
    shot_types = ['shot-on-goal', 'missed-shot', 'goal']
    shots = df[df['event'].isin(shot_types)].head(5)
    
    print("\nTesting Adjustments on Sample Shots:")
    for idx, row in shots.iterrows():
        gid = row.get('game_id')
        x = row.get('x')
        y = row.get('y')
        
        home_team = game_map.get(gid, 'Unknown')
        print(f"  Game {gid} | Home: '{home_team}' | Raw: ({x}, {y})")
        
        xa, ya = arena_adjustments.adjust_shot(x, y, home_team, season)
        print(f"    -> Adj: ({xa}, {ya}) {'[CHANGED]' if xa != x else '[SAME]'}")
        
    # Check if adjustments loaded
    adj_data = arena_adjustments.load_adjustments()
    print(f"\nLoaded Adjustments Keys: {list(adj_data.keys())}")
    if str(season) in adj_data:
        print(f"  Season {season} in adjustments: Yes")
        s_data = adj_data[str(season)]
        print(f"  Arenas in season: {list(s_data.keys())[:5]}...")
    else:
        print(f"  Season {season} in adjustments: NO (Fallback logic applies?)")

if __name__ == "__main__":
    debug_season("20232024")
