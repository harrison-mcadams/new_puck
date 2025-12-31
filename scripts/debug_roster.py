import sys
import os
import json

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck.nhl_api import get_game_feed

# Use a known game ID from the previous run output (2025020583)
game_id = 2025020583

print(f"Fetching feed for {game_id}...")
feed = get_game_feed(game_id)

if not feed:
    print("No feed found.")
    sys.exit()

print("Top Level Keys:", feed.keys())

# Check for Roster Locations
if 'playerByTeam' in feed:
    print("\n--- playerByTeam ---")
    pbt = feed['playerByTeam']
    for side in ['homeTeam', 'awayTeam']:
        print(f"\nSide: {side}")
        team_data = pbt.get(side, {})
        print("Keys:", team_data.keys())
        
        # Check potential list keys
        for key in ['roster', 'rosterSpots', 'players', 'forwards', 'defense', 'goalies']:
            if key in team_data:
                print(f"  Found '{key}' with {len(team_data[key])} items.")
                if len(team_data[key]) > 0:
                    sample = team_data[key][0]
                    print(f"  Sample item keys: {sample.keys()}")
                    if 'firstName' in sample: print(f"    firstName: {sample['firstName']}")
                    if 'name' in sample: print(f"    name: {sample['name']}")

if 'gameData' in feed:
    print("\n--- gameData ---")
    gd = feed['gameData']
    if 'players' in gd:
        print(f"Found {len(gd['players'])} players in gameData.")
        # sample one
        pid, pdata = next(iter(gd['players'].items()))
        print(f"Sample player data keys: {pdata.keys()}")
        print(f"Sample player: {pdata}")

if 'rosterSpots' in feed:
    print("\n--- Top Level rosterSpots ---")
    print(f"Count: {len(feed['rosterSpots'])}")
