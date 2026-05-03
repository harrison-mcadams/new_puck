import os
import sys
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import timing

gid = 2025020352 # One of the "bad" games from the log
season = '20252026'

print(f"Testing Game {gid}...")

# Test 1: Full condition
cond1 = {'game_state': ['5v5'], 'is_net_empty': [0]}
res1 = timing.compute_intervals_for_game(gid, cond1, season=season, verbose=True)
print(f"Result 1 (Full): {res1['intersection_seconds']}s")

# Test 2: Only game_state
cond2 = {'game_state': ['5v5']}
res2 = timing.compute_intervals_for_game(gid, cond2, season=season, verbose=True)
print(f"Result 2 (GS only): {res2['intersection_seconds']}s")

# Test 3: Only is_net_empty
cond3 = {'is_net_empty': [0]}
res3 = timing.compute_intervals_for_game(gid, cond3, season=season, verbose=True)
print(f"Result 3 (Net empty only): {res3['intersection_seconds']}s")
