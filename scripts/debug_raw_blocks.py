
import sys
from pathlib import Path
# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from puck import parse, nhl_api

# Select a recent game to debug (e.g., from 2025 season)
# We need a game ID. Let's use 2024020001 (Season Opener usually exists) or search for one.
# For now, let's just use the load_data() function to find a game_id from the processed data
# and then fetch its raw feed.

print("Loading sample processed data to find a game ID...")
try:
    df_sample = pd.read_csv('data/20252026/20252026_df.csv', nrows=1000)
    # Find a game with blocked shots
    game_id = df_sample[df_sample['event'] == 'blocked-shot']['game_id'].iloc[0]
    print(f"Selected Game ID: {game_id}")
except Exception as e:
    print(f"Could not find game from processed data: {e}")
    # Fallback to a known ID if possible, or exit
    sys.exit(1)

# Fetch RAW game feed
print(f"Fetching raw feed for {game_id}...")
# parse._scrape uses nhl_api internally.
# We want to use parse._game(game_id, season) but catch the intermediate 'plays'.
# Actually, parse._game calls nhl_api.get_game_feed(game_id).
# Let's inspect that raw feed.

import json
raw_feed = nhl_api.get_game_feed(game_id)
# This returns the full dict.

# Look for blocked shots in 'plays'
print("Inspecting Raw Blocked Shots...")

# Structure changes between API versions. Assuming standard NHL API v1 or similar?
# Using 'plays' list?
if 'plays' in raw_feed:
    plays = raw_feed['plays']
elif 'liveData' in raw_feed and 'plays' in raw_feed['liveData']:
    plays = raw_feed['liveData']['plays']['allPlays']
else:
    # Try finding plays list
    print("Could not locate 'plays' in feed keys: ", raw_feed.keys())
    plays = []

blocks_found = 0
for p in plays:
    # Check event type
    evt = p.get('typeDescKey', '') or p.get('result', {}).get('event', '')
    if evt == 'blocked-shot':
        blocks_found += 1
        print("\n--- Raw Blocked Shot ---")
        # Print relevant fields
        # Coordinates
        coords = p.get('details', {})
        if not coords: coords = p.get('coordinates', {})
        
        # Team / Owner
        # Usually details has 'blockingPlayerId', 'shootingPlayerId'
        # Or 'teamId' is the owner (Blocker)
        
        print(f"Raw Coords: {coords}", flush=True)
        
        # Event Owner
        owner = p.get('details', {}).get('eventOwnerTeamId') # v1 style?
        if not owner: owner = p.get('team', {}).get('id')
        print(f"Event Owner Team ID: {owner}", flush=True)
        
        # Context
        period = p.get('periodDescriptor', {}).get('number')
        time = p.get('timeInPeriod')
        print(f"Period: {period}, Time: {time}", flush=True)
        
        if blocks_found >= 1: # Just find ONE block
            break

print("\n--- Compare with Parsed Output ---")
# Now run parse._game on this single game to see what our logic did
# Now run parse._game on this single game to see what our logic did
print("\n--- Compare with Parsed Output ---", flush=True)
try:
    print("Calling parse._game...", flush=True)
    df_parsed = parse._game(raw_feed)
    print(f"Parsed Shape: {df_parsed.shape}", flush=True)
    
    if not df_parsed.empty:
        print("Columns found:", df_parsed.columns.tolist(), flush=True)
        df_blocks = df_parsed[df_parsed['event'] == 'blocked-shot'].head(5)
        if not df_blocks.empty:
             # Print specific fields robustly
             print("First Blocked Shot Parsed Details:", flush=True)
             row = df_blocks.iloc[0]
             print(f"  Event: {row.get('event')}", flush=True)
             print(f"  Team: {row.get('event_team')}", flush=True)
             print(f"  X: {row.get('x')}", flush=True)
             print(f"  Y: {row.get('y')}", flush=True)
             print(f"  Period: {row.get('period')}", flush=True)
             print(f"  Home Def Side: {row.get('home_team_defending_side')}", flush=True)
             # Try to print home/away if available (cols might differ)
             if 'home_team' in df_parsed.columns:
                 print(f"  Home Team: {row.get('home_team')}", flush=True)
        else:
             print("No blocked shots found in parsed DF!", flush=True)
             print("Events found:", df_parsed['event'].unique(), flush=True)
    else:
        print("Parsed DF is empty!", flush=True)

except Exception as e:
    print(f"Parsing failed: {e}", flush=True)
    import traceback
    traceback.print_exc()
