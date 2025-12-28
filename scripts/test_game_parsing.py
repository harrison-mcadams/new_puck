
import sys
import os
import pandas as pd
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import nhl_api, parse

def test_game_parse(game_id):
    print(f"Fetching game {game_id}...")
    feed = nhl_api.get_game_feed(game_id)
    print("Parsing game...")
    df = parse._game(feed)
    
    # Filter for the specific shot
    # x=73, y=9, period=1
    mask = (df['period'] == 1) & (df['event'] == 'shot-on-goal')
    # Filter by coords roughly
    mask &= (df['x'] - 73.0).abs() < 1.0
    mask &= (df['y'] - 9.0).abs() < 1.0
    
    rows = df[mask]
    print(f"Found {len(rows)} matching rows in parsed output.")
    if not rows.empty:
        print(rows[['event', 'period', 'period_time', 'time_elapsed_in_period_s', 'total_time_elapsed_s']].to_string())
    else:
        print("No matching rows found in parsed output.")

if __name__ == "__main__":
    test_game_parse(2025020181)
