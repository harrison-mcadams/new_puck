import pickle
import numpy as np
from pathlib import Path

path = Path("analysis/mixed_effects_heatmaps_20252026/team_grids.pkl")

if path.exists():
    with open(path, 'rb') as f:
        grids = pickle.load(f)
    
    keys = list(grids.keys())
    print("Keys found:", keys[:10])
    print("Total Keys:", len(keys))
    
    if 'League' in grids:
        print("League key found!")
    else:
        print("League key NOT found. Will need to Aggregate.")
        
    # Check structure of one team
    if keys:
        t = keys[0]
        print(f"Structure for {t}: {grids[t].keys()}")
else:
    print(f"File not found: {path}")
