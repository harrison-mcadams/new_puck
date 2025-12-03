import nhl_api
import parse
import pandas as pd
import logging

# Configure logging to see details
logging.basicConfig(level=logging.INFO)

game_id = 2025020318

print(f"--- Investigating Game {game_id} ---")

# 1. Check Game Status/Metadata
try:
    feed = nhl_api.get_game_feed(game_id)
    if feed:
        game_data = feed.get('gameData', {})
        status = game_data.get('status', {})
        print(f"Game Status: {status}")
        datetime = game_data.get('datetime', {})
        print(f"Game DateTime: {datetime}")
    else:
        print("Could not fetch game feed.")
except Exception as e:
    print(f"Error fetching game feed: {e}")

# 2. Check Standard API Shifts
print("\n--- Standard API Shifts ---")
try:
    shifts_res = nhl_api.get_shifts(game_id)
    print(f"API Response Type: {type(shifts_res)}")
    if isinstance(shifts_res, dict):
        data = shifts_res.get('data', [])
        print(f"API Data Length: {len(data)}")
    else:
        print(f"API Response: {shifts_res}")
except Exception as e:
    print(f"Standard API failed: {e}")

# 3. Check HTML Fallback
print("\n--- HTML Fallback Shifts ---")
try:
    html_shifts = nhl_api.get_shifts_from_nhl_html(game_id, debug=True)
    print(f"HTML Fallback Result Type: {type(html_shifts)}")
    if isinstance(html_shifts, list):
        print(f"HTML Fallback List Length: {len(html_shifts)}")
        if len(html_shifts) > 0:
            print(f"First item: {html_shifts[0]}")
            df = pd.DataFrame(html_shifts)
            print(f"DataFrame Shape: {df.shape}")
            print(df.head())
    elif isinstance(html_shifts, dict):
         print(f"HTML Fallback Dict Keys: {html_shifts.keys()}")
    else:
        print("HTML Fallback returned empty or invalid type.")
except Exception as e:
    print(f"HTML Fallback failed: {e}")
