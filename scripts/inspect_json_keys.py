
import json
import sys
from pathlib import Path

def main():
    path = Path("data/20252026/game_2025020670.json")
    if not path.exists():
        print(f"File not found: {path}")
        return

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        roster = data.get('rosterSpots', [])
        print(f"Roster spots found: {len(roster)}")
        
        if roster:
            first = roster[0]
            print("\nFirst Roster Spot Keys:")
            print(list(first.keys()))
            print(f"Sample Position: {first.get('positionCode')}")
            print(f"Sample Shoots: {first.get('shootsCatches')}")
            
        print("\nChecking gameData.players...")
        players = data.get('gameData', {}).get('players', {})
        if players:
             # keys are IDs usually
             first_id = next(iter(players))
             print(f"First Player ID: {first_id}")
             pdata = players[first_id]
             print("Player Keys:", list(pdata.keys()))
             print(f"Sample Shoots: {pdata.get('shootsCatches')}")
             # check nested?
        else:
             print("No gameData.players found.")
            
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
