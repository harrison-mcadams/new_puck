
import pandas as pd
import numpy as np
import os
import sys
import joblib
from scipy.interpolate import interp1d

# Add project root to path
sys.path.append(os.getcwd())
from puck import fit_xgs, rink

def calculate_distance(x, y):
    dists = []
    for ix, iy in zip(x, y):
        nx_ref = 89 if ix >= 0 else -89
        d = np.sqrt((ix - nx_ref)**2 + iy**2)
        dists.append(d)
    return np.array(dists)

def get_cdf(data, num_bins=1000):
    data = np.sort(data[~np.isnan(data)])
    n = len(data)
    if n == 0: return None, None
    p_grid = np.linspace(0, 1, min(num_bins, n))
    x_grid = np.quantile(data, p_grid)
    cdf_func = interp1d(x_grid, p_grid, kind='linear', bounds_error=False, fill_value=(0, 1))
    icdf_func = interp1d(p_grid, x_grid, kind='linear', bounds_error=False, fill_value="extrapolate")
    return cdf_func, icdf_func

def main():
    print("--- Generating Self-Consistent Mapping (Summary-Only) ---")
    summary_path = 'analysis/blocked_shots/blocked_shots_summary_batch.csv'
    df = pd.read_csv(summary_path)
    
    # FILTER BY SCORE > 0.4 (The user's ground truth)
    df = df[df['score'] > 0.4].copy()
    print(f"Using {len(df)} records for mapping.")

    # Block distance from net
    # Reconstructed: dist_net_origin - dist_to_blocker? 
    # Or just use the actual block dist if we had it.
    # In the summary, d_shooter is distance from shooter to net.
    # d_blocker is distance from shooter to blocker.
    # So d_net_block = d_shooter - d_blocker (assuming straight line)
    d_origins = calculate_distance(df['x'], df['y'])
    d_blocks = d_origins - df['distance_to_blocker']
    
    mappings = {}
    for role in ['F', 'D']:
        mask = (df['shooter_role'] == role)
        d_org_role = d_origins[mask]
        d_blk_role = d_blocks[mask]
        
        print(f"  Role {role}: {len(d_org_role)} samples. Mean Org: {d_org_role.mean():.1f}, Mean Blk: {d_blk_role.mean():.1f}")
        
        cdf_blk, _ = get_cdf(d_blk_role)
        _, icdf_org = get_cdf(d_org_role)
        
        mappings[role] = {'cdf_block': cdf_blk, 'icdf_origin': icdf_org}
    
    # Global fallback
    cdf_blk_all, _ = get_cdf(d_blocks)
    _, icdf_org_all = get_cdf(d_origins)
    mappings['global'] = {'cdf_block': cdf_blk_all, 'icdf_origin': icdf_org_all}
    
    out_path = 'puck/data/cdf_mappings.joblib'
    joblib.dump(mappings, out_path)
    print(f"Saved to {out_path}")

    # Sanity check
    print("\nSanity Check (Self-Consistent):")
    for d in [5, 10, 15, 20]:
        p = cdf_blk_all(d)
        o = icdf_org_all(p)
        print(f"  Block {d:2d} ft -> Origin {o:5.1f} ft")

if __name__ == "__main__":
    main()
