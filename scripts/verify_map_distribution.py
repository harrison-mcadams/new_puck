import numpy as np
import os

def check_map(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    
    data = np.load(path)
    # The grid is 100x200 (Y, X) usually, or some similar shape depending on res
    # Let's assume the maps are oriented such that the horizontal axis is the second dimension.
    # Map orientation: Offense Left (-100 to 0 on rink X, but plotting usually maps it)
    # In relative_combined.npy, it's a 2D grid.
    
    mid = data.shape[1] // 2
    left_half = data[:, :mid]
    right_half = data[:, mid:]
    
    left_mean = np.nanmean(left_half)
    right_mean = np.nanmean(right_half)
    
    print(f"Map: {os.path.basename(path)}")
    print(f"  Shape: {data.shape}")
    print(f"  Left Mean (Offense): {left_mean:.6f}")
    print(f"  Right Mean (Defense): {right_mean:.6f}")
    
    # Check for near-zero values
    if abs(right_mean) < 1e-6:
        print("  WARNING: Right half (Defense) appears near-zero!")
    else:
        print("  OK: Right half (Defense) has non-zero data.")

# Check a few teams
base_dir = r'c:\Users\harri\Desktop\new_puck\analysis\league\20252026\5v5'
check_map(os.path.join(base_dir, 'ANA_relative_combined.npy'))
check_map(os.path.join(base_dir, 'PHI_relative_combined.npy'))
check_map(os.path.join(base_dir, 'PIT_relative_combined.npy'))
check_map(os.path.join(base_dir, 'VGK_relative_combined.npy'))
check_map(os.path.join(base_dir, 'COL_relative_combined.npy'))
check_map(os.path.join(base_dir, 'EDM_relative_combined.npy'))
check_map(os.path.join(base_dir, 'baseline.npy'))
