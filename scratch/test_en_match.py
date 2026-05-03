import os
import sys
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import timing

gid = 2025021109
season = '20252026'

print(f"Testing Game {gid}...")

# Test 1: Our standard (no empty net)
cond1 = {'game_state': ['5v5'], 'is_net_empty': [0]}
res1 = timing.compute_intervals_for_game(gid, cond1, season=season)
print(f"Our 5v5 (no EN): {res1['intersection_seconds']}s")

# Test 2: Include empty net
cond2 = {'game_state': ['5v5'], 'is_net_empty': [0, 1]}
res2 = timing.compute_intervals_for_game(gid, cond2, season=season)
print(f"Our 5v5 (with EN): {res2['intersection_seconds']}s")

# Test 3: GS reported (manual check of the log showed 2971s)
# Let's see if Test 2 matches GS exactly.
