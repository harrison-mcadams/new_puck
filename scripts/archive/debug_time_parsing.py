
import sys
import os
import json
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import nhl_api

def debug_game(game_id):
    print(f"Loading raw data for game {game_id}...")
    # Try to load from disk first if available (mimicking standard flow)
    # But for debugging, fetching fresh is fine if needed.
    
    # We will use nhl_api logic to fetch/load
    data = nhl_api.get_game_feed(game_id)
    
    if not data:
        print("Failed to load data.")
        return

    plays = data.get('plays', [])
    print(f"Found {len(plays)} plays.")
    
    for i, p in enumerate(plays):
        # We are looking for shots where time parsing might fail
        ev_type = p.get('typeDescKey') or (p.get('type') or {}).get('description') or p.get('typeCode')
        if ev_type in ['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal']:
            
            time_remaining = p.get('timeRemaining')
            time_in_period = p.get('timeInPeriod')
            period = p.get('periodDescriptor', {}).get('number') if isinstance(p.get('periodDescriptor'), dict) else p.get('period')
            
            # Recreate the check
            per_len = 1200 # approx
            
            # Match the specific event from verify script: x=73.0, y=9.0, period=1
            curr_x = p.get('details', {}).get('xCoord') if isinstance(p.get('details'), dict) else p.get('details', {}).get('x')
            curr_y = p.get('details', {}).get('yCoord') if isinstance(p.get('details'), dict) else p.get('details', {}).get('y')
            
            # Print first shot found
            if 'shot' in ev_type or 'goal' in ev_type:
                 print(f"FOUND SHOT: {ev_type}")
                 print(f"TimeRemaining: {time_remaining}")
                 print(f"TimeInPeriod: {time_in_period}")
                 print(f"PeriodDescriptor: {p.get('periodDescriptor')}")
                 print(f"Raw Play: {json.dumps(p, indent=2)}")
                 break

if __name__ == "__main__":
    # From previous failure output
    debug_game(2025020460)
