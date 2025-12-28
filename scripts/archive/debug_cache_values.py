
import os
import sys
import numpy as np
import json
import glob

# Path to cache
CACHE_DIR = "data/cache/20252026/partials"

def check_cache():
    if not os.path.exists(CACHE_DIR):
        print(f"Cache dir {CACHE_DIR} does not exist!")
        return

    files = glob.glob(os.path.join(CACHE_DIR, "*.npz"))
    print(f"Found {len(files)} cache files.")
    
    total_xg = 0.0
    total_goals = 0
    
    scanned = 0
    for f in files[:50]: # Scan first 50
        try:
            with np.load(f, allow_pickle=True) as data:
                if 'empty' in data: continue
                
                keys = list(data.keys())
                for k in keys:
                    if k.endswith('_stats'):
                        # stats are stored as JSON string inside 0-d array usually
                        item = data[k]
                        if item.ndim == 0:
                            s = json.loads(str(item))
                        else:
                            s = json.loads(str(item[0]))
                            
                        # s has 'team_xgs', 'team_goals' etc per game
                        total_xg += s.get('team_xgs', 0.0)
                        total_goals += s.get('team_goals', 0)
                        
                scanned += 1
        except Exception as e:
            print(f"Error reading {f}: {e}")

    print(f"\n--- Cache Check (First {scanned} files) ---")
    print(f"Total xG: {total_xg:.2f}")
    print(f"Total Goals: {total_goals}")
    if total_goals > 0:
        print(f"Ratio (xG/Goals): {total_xg/total_goals:.3f}")
    else:
        print("No goals found.")

if __name__ == "__main__":
    check_cache()
