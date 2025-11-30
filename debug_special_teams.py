
import numpy as np
import os
import matplotlib.pyplot as plt

def inspect_map(path, name):
    if not os.path.exists(path):
        print(f"{name}: File not found at {path}")
        return

    data = np.load(path)
    print(f"--- {name} ---")
    print(f"Shape: {data.shape}")
    
    mid = data.shape[1] // 2
    left = data[:, :mid]
    right = data[:, mid:]
    
    print(f"Left (Offense) - Mean: {np.nanmean(left):.4f}, Max: {np.nanmax(left):.4f}, Sum: {np.nansum(left):.4f}")
    print(f"Right (Defense) - Mean: {np.nanmean(right):.4f}, Max: {np.nanmax(right):.4f}, Sum: {np.nansum(right):.4f}")
    
    # Check for NaNs
    print(f"NaNs: {np.isnan(data).sum()} / {data.size}")

base_dir = 'static/league/20252026'
team = 'ANA'

inspect_map(os.path.join(base_dir, '5v4', f'{team}_relative_combined.npy'), '5v4 (PP)')
inspect_map(os.path.join(base_dir, '4v5', f'{team}_relative_combined.npy'), '4v5 (PK)')
