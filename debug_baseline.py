
import numpy as np
import os

def inspect_map(path, name):
    if not os.path.exists(path):
        print(f"{name}: File not found at {path}")
        return

    data = np.load(path)
    print(f"--- {name} ---")
    print(f"Shape: {data.shape}")
    print(f"Mean: {np.nanmean(data):.4f}, Max: {np.nanmax(data):.4f}, Sum: {np.nansum(data):.4f}")

base_dir = 'static/league/20252026'

# Check 4v5 Baseline Right (League PK Defense)
inspect_map(os.path.join(base_dir, '4v5', '20252026_league_baseline_right.npy'), '4v5 Baseline Right (PK Defense)')

# Check 4v5 Baseline Left (League SH Offense)
inspect_map(os.path.join(base_dir, '4v5', '20252026_league_baseline.npy'), '4v5 Baseline Left (SH Offense)')
