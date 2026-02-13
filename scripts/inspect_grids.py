import pickle
import numpy as np

path = "analysis/mixed_effects_heatmaps_20252026/team_grids.pkl"
try:
    with open(path, 'rb') as f:
        grids = pickle.load(f)
    
    print("Keys in grids:", list(grids.keys())[:5])
    
    nyr_5v5 = grids['NYR']['5v5']
    print("\n--- NYR 5v5 Grid Stats ---")
    print(f"Grid For Shape: {nyr_5v5['grid_for'].shape}")
    print(f"Grid For Sum: {nyr_5v5['grid_for'].sum()}")
    print(f"Grid For Max: {nyr_5v5['grid_for'].max()}")
    
    edm_5v5 = grids['EDM']['5v5']
    print("\n--- EDM 5v5 Grid Stats ---")
    print(f"Grid Against Shape: {edm_5v5['grid_against'].shape}")
    print(f"Grid Against Sum: {edm_5v5['grid_against'].sum()}")
    
    # Debug the function
    import sys
    import os
    scripts_dir = os.path.join(os.getcwd(), 'scripts')
    if scripts_dir not in sys.path:
        sys.path.append(scripts_dir)
    from matchup import get_matchup_density
    
    print("\n--- Testing get_matchup_density ---")

    dens = get_matchup_density(nyr_5v5['grid_for'], edm_5v5['grid_against'], None, None)
    print(f"Result Density Sum: {dens.sum()}")
    print(f"Result Density Max: {dens.max()}")

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Error: {e}")

