import sys
import os
import json
# Add project root to path to use puck package if needed, 
# but for now standard request or using existing nhl_api logic is best.
# I'll use the 'puck' package logic if I can import it, or just mock it.
# Actually, I'll use the 'puck.nhl_api' module if available.

sys.path.append(os.getcwd())
try:
    from puck import nhl_api
except ImportError:
    print("Could not import puck.nhl_api")
    sys.exit(1)

import pprint

def check_pbp():
    game_id = 2024020202
    # Use get_game_feed
    feed = nhl_api.get_game_feed(game_id)
    
    if not feed:
        print("Failed to get feed.")
        return

    plays = feed.get('plays', [])
    
    goal_event = next((p for p in plays if str(p.get('eventId')) == '328'), None)
    block_event = next((p for p in plays if str(p.get('eventId')) == '325'), None)
    
    if goal_event:
        print("--- Goal Event (328) Keys ---")
        print(goal_event.keys())
        # Check specific time keys
        print(f"timeInPeriod: {goal_event.get('timeInPeriod')}")
        print(f"timeRemaining: {goal_event.get('timeRemaining')}")
        # Look for timestamp
        # Common locations: 'periodDescriptor', 'about'? No, root.
        # Check for ANY key with 'time' or 'date'
        import pprint
        pprint.pprint(goal_event)
        
    if block_event:
        print("\n--- Block Event (325) Keys ---")
        # Check if it has the same timestamp key
        pprint.pprint(block_event)

if __name__ == "__main__":
    check_pbp()
