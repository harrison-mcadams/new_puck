import numpy as np
import os
import sys

# Find a valid cache file
cache_dir = "data/cache/20252026/partials"
files = [f for f in os.listdir(cache_dir) if f.endswith('5v5.npz') and 'empty' not in f]

if not files:
    print("No cache files found.")
    sys.exit(1)

fpath = os.path.join(cache_dir, files[0])
print(f"Inspecting {fpath}")

try:
    with np.load(fpath, allow_pickle=True) as data:
        print("Keys:", list(data.keys()))
        for k in data.keys():
            if k.endswith('_stats'):
                print(f"\n--- Key: {k} ---")
                val = data[k]
                print(f"Type: {type(val)}")
                print(f"Shape: {val.shape}")
                print(f"Dtype: {val.dtype}")
                
                # Try to extract the item
                try:
                    item = val.item()
                    print(f"Item Type: {type(item)}")
                    print(f"Item Content: {item}")
                except Exception as e:
                    print(f"Item extraction failed: {e}")
                    
            if k.endswith('_grid_team'):
                 print(f"\n--- Key: {k} (Grid) ---")
                 val = data[k]
                 print(f"Shape: {val.shape}")
                 print(f"Sum: {np.sum(val)}")
                 
except Exception as e:
    print(f"Failed to load: {e}")
