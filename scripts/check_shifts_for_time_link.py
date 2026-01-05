import sys
import os
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.getcwd())
try:
    from puck import nhl_api
except ImportError:
    print("Could not import puck.nhl_api")
    sys.exit(1)

def check_shifts():
    game_id = 2024020202
    print(f"Fetching shifts for game {game_id}...")
    
    # Force refresh to ensure we get headers/fresh data if needed, 
    # though standard cache is fine.
    data = nhl_api.get_shifts(game_id)
    
    shifts = data.get('data', []) # Standard shape often has 'data' key
    # If not in 'data', check 'all_shifts' from our wrapper
    if not shifts:
        shifts = data.get('all_shifts', [])
        
    if not shifts:
        print("No shifts found.")
        # Debug structure
        print(f"Keys in response: {list(data.keys())}")
        return

    print(f"Found {len(shifts)} shifts.")
    
    # Inspect first shift for keys
    first = shifts[0]
    print(f"\nSample Shift Keys: {list(first.keys())}")
    
    # Check for TIME keys
    # We need both "Game Time" (startTime) and "Wall Time" (startTimeUTC/start)
    
    sample_with_time = next((s for s in shifts if 'startTime' in s), None)
    
    if sample_with_time:
        print("\n--- Shift Time Data ---")
        for k in ['startTime', 'endTime', 'duration', 'period', 'firstName', 'lastName']:
             if k in sample_with_time:
                 print(f"{k}: {sample_with_time[k]}")
                 
        # Look for wall clock
        # Often plain 'start' or 'end' might be something different, or hidden in header?
        # Sometimes it is NOT in the per-shift data but in the game info?
        pass
    else:
        print("No 'startTime' key found in shifts.")
        
    print("\nFull Sample:")
    print(json.dumps(first, indent=2))

if __name__ == "__main__":
    check_shifts()
