
import numpy as np
import sys
import os

# Inspect a specific cache file
# Using one that is reasonably large
path = "data/cache/20252026/partials/2025020483_5v5.npz"

if not os.path.exists(path):
    print(f"File not found: {path}")
    sys.exit(1)

print(f"Loading {path}...")
with np.load(path, allow_pickle=True) as data:
    print(f"Keys: {list(data.keys())}")
    
    if 'processed' in data:
        print(f"Processed: {data['processed']}")
    if 'empty' in data:
        print(f"Empty: {data['empty']}")
        
    # Find a team grid
    team_keys = [k for k in data.keys() if k.endswith('_grid_team')]
    for k in team_keys:
        grid = data[k]
        print(f"\n--- {k} ---")
        print(f"Shape: {grid.shape}")
        print(f"Dtype: {grid.dtype}")
        print(f"Min: {np.min(grid)}")
        print(f"Max: {np.max(grid)}")
        print(f"Mean: {np.mean(grid)}")
        print(f"Non-zero count: {np.count_nonzero(grid)}")
        print(f"Sum: {np.sum(grid)}")
        
        # Stats
        stats_key = k.replace('_grid_team', '_stats')
        if stats_key in data:
             print(f"Stats: {data[stats_key]}")

