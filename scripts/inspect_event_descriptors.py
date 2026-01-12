
import json
import pandas as pd
from pathlib import Path
import sys

# Project root setup
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / 'data'

def inspect_2014_sample():
    print("\n--- Inspecting 2014-2015 Season Data ---")
    path_2014 = DATA_DIR / '20142015' / '20142015_game_feeds.json'
    
    if not path_2014.exists():
        print(f"Error: {path_2014} not found.")
        return

    try:
        print(f"Loading {path_2014.name}...")
        with open(path_2014, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        games_list = []
        if isinstance(data, list):
            games_list = data
        elif isinstance(data, dict):
            games_list = list(data.values())
            
        print(f"Loaded {len(games_list)} games.")
        
        # Search for a valid game with plays
        valid_game = None
        plays = []
        
        print("Searching for a valid game with plays...")
        for i, g in enumerate(games_list):
            gid = g.get('gamePk') or g.get('id')
            
            # Try to find plays
            curr_plays = []
            if 'liveData' in g and 'plays' in g['liveData']:
                curr_plays = g['liveData']['plays']['allPlays']
            elif 'plays' in g:
                curr_plays = g['plays']
                
            if curr_plays:
                valid_game = g
                plays = curr_plays
                print(f"Found valid game at index {i}: ID {gid} with {len(plays)} plays.")
                break
        
        if valid_game:
            print_blocked_shots(plays, "2014-2015")
        else:
            print("Could not find any 2014 game with parsed plays.")
        
    except Exception as e:
        print(f"Failed to inspect 2014 data: {e}")

def inspect_2025_sample():
    print("\n--- Inspecting 2025-2026 Season Data (PHI vs ANA) ---")
    files = list(DATA_DIR.glob('**/game_2025020670.json'))
    if files:
        path_2025 = files[0]
    else:
        path_2025 = DATA_DIR / '20252026' / 'game_2025020670.json'
    
    if not path_2025.exists():
        print(f"Error: {path_2025} not found.")
        return

    print(f"Loading {path_2025.name}...")
    try:
        with open(path_2025, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        plays = []
        if 'plays' in data:
            plays = data['plays']
        elif 'playByPlay' in data:
             plays = data['playByPlay']
        elif 'liveData' in data:
            plays = data['liveData']['plays']['allPlays']
            
        print_blocked_shots(plays, "2025-2026")

    except Exception as e:
        print(f"Failed to inspect 2025 data: {e}")

def print_blocked_shots(plays, label):
    print(f"\n{label} Blocked Shot Samples:")
    count = 0
    for play in plays:
        # Check event type
        event_type = ""
        description = ""
        
        # Normalized extraction
        if 'result' in play: # Old Style
            event_type = play['result'].get('event')
            description = play['result'].get('description')
        elif 'typeDescKey' in play: # New Style
            event_type = play.get('typeDescKey')
        
        # Filter for blocks
        if event_type in ['BLOCKED_SHOT', 'blocked-shot']:
            

            print(f"\n[Sample {count+1}]")
            print(f"Event Type: {event_type}")
            
            # Attributed Team
            curr_team_id = "N/A"
            if 'team' in play:
                curr_team_id = play['team'].get('id')
            elif 'details' in play and 'eventOwnerTeamId' in play['details']:
                curr_team_id = play['details']['eventOwnerTeamId']
            
            print(f"Attributed Team ID: {curr_team_id}")
            
            # Details
            details = play.get('details', {})
            blocker_id = details.get('blockingPlayerId')
            shooter_id = details.get('shootingPlayerId')
            
            print(f"Blocker ID: {blocker_id}")
            print(f"Shooter ID: {shooter_id}")
            print(f"Raw Details: {details}")
            
            print("-" * 40)
            
            count += 1
            if count >= 3:
                break
    
    if count == 0:
        print("  No blocked shots found in this game.")

if __name__ == "__main__":
    inspect_2014_sample()
    inspect_2025_sample()
