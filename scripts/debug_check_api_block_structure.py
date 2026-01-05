import sys
import os
sys.path.append(os.getcwd())
from puck import nhl_api
import json

def check_structure():
    game_id = 2024020202
    print(f"Fetching feed for {game_id}...")
    feed = nhl_api.get_game_feed(game_id)
    
    plays = feed.get('plays', [])
    print(f"Found {len(plays)} plays.")
    
    # Find first blocked shot
    block_play = None
    for p in plays:
        if p.get('typeDescKey') == 'blocked-shot':
            block_play = p
            break
            
    if block_play:
        print("\n--- Blocked Shot Event Structure ---")
        print(json.dumps(block_play, indent=2))
        
        details = block_play.get('details', {})
        print("\nDetails Keys:", details.keys())
        if 'blockingPlayerId' in details:
            print("Found: blockingPlayerId")
        if 'shootingPlayerId' in details:
            print("Found: shootingPlayerId")
    else:
        print("No blocked shots found in this game.")

if __name__ == "__main__":
    check_structure()
