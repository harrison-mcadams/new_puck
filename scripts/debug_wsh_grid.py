import os
import numpy as np
import sys
sys.path.append(os.getcwd())
from puck import config

season = '20252026'
cache_dir = os.path.join(config.get_cache_dir(season), 'partials')
files = sorted([f for f in os.listdir(cache_dir) if f.endswith('_5v5.npz')])

print(f"Found {len(files)} cache files.")
wsh_id = 15 # Capitals ID
wsh_grid_sum = 0
wsh_seconds = 0
found_games = 0

for f in files:
    path = os.path.join(cache_dir, f)
    try:
        data = np.load(path)
        keys = list(data.keys())
        # Check team keys
        k_grid = f"team_{wsh_id}_grid_team"
        k_stats = f"team_{wsh_id}_stats"
        
        if k_grid in data:
            g = data[k_grid]
            wsh_grid_sum += g
            found_games += 1
            
        if k_stats in data:
            s = data[k_stats].item()
            wsh_seconds += s.get('team_seconds', 0)
            
    except Exception as e:
        print(f"Error {f}: {e}")

print(f"WSH Games Found: {found_games}")
if isinstance(wsh_grid_sum, (int, float)) and wsh_grid_sum == 0:
    print("WSH Grid Sum is 0!")
else:
    print(f"WSH Grid Sum Max: {np.max(wsh_grid_sum)}")
    print(f"WSH Grid Sum Min: {np.min(wsh_grid_sum)}")
    print(f"WSH Grid Sum Mean: {np.mean(wsh_grid_sum)}")

print(f"WSH Total Seconds: {wsh_seconds}")

# Check League Grid logic (sum of all team grids?)
# In loop we summed specific team.
