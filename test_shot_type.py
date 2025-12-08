
import sys
import os
sys.path.append(os.getcwd())
from puck import parse, nhl_api

def test_parse():
    # Pick a game ID (standard format 202402xxxx for 2024-2025 regular season)
    # Using a 2024-2025 game as example, or 20252026 if synthetic future data
    game_id = 2024020001 
    
    print(f"Fetching game {game_id}...")
    feed = nhl_api.get_game_feed(game_id)
    if not feed:
        print("Failed to fetch feed")
        return

    print("Parsing game...")
    df = parse._game(feed)
    
    if df.empty:
        print("No events found in DataFrame")
        return
        
    if 'shot_type' in df.columns:
        print("SUCCESS: 'shot_type' column found!")
        print("Sample values:")
        print(df['shot_type'].unique())
        print(df[['event', 'shot_type']].head())
    else:
        print("FAILURE: 'shot_type' column NOT found.")
        print("Columns:", df.columns.tolist())

if __name__ == "__main__":
    test_parse()
