
import pandas as pd
import numpy as np
import os
import sys
import joblib
from scipy.interpolate import interp1d

# Add project root to path
sys.path.append(os.getcwd())
from puck import fit_xgs, correction, rink

def calculate_distance(x, y):
    dists = []
    for ix, iy in zip(x, y):
        nx_ref = 89 if ix >= 0 else -89
        d = np.sqrt((ix - nx_ref)**2 + iy**2)
        dists.append(d)
    return np.array(dists)

def get_cdf(data, num_bins=1000):
    # Add dithering/jitter to break quantization steps
    # NHL coords are often integers. This creates flat spots in Inverse CDF.
    # +/- 0.25 ft is within measurement error but ensures unique values.
    # Use deterministic seed within function if needed, but random is fine for large N.
    jitter = np.random.uniform(-0.25, 0.25, size=len(data))
    data = data + jitter

    data = np.sort(data[~np.isnan(data)])
    n = len(data)
    if n == 0: return None, None
    p_grid = np.linspace(0, 1, min(num_bins, n))
    x_grid = np.quantile(data, p_grid)
    cdf_func = interp1d(x_grid, p_grid, kind='linear', bounds_error=False, fill_value=(0, 1))
    icdf_func = interp1d(p_grid, x_grid, kind='linear', bounds_error=False, fill_value="extrapolate")
    return cdf_func, icdf_func

def main():
    print("--- Generating Production CDF Mappings (PBP Baseline) ---")
    all_blocks = []
    all_origins_f = []
    all_origins_d = []
    
    seasons = ['20232024', '20242025', '20252026']
    for s_str in seasons:
        path = f"data/{s_str}/{s_str}_df.csv"
        if not os.path.exists(path): continue
        print(f"  Loading {s_str}...")
        df = pd.read_csv(path)
        
        # Blocks (PBP)
        blocks = df[df['event'] == 'blocked-shot'].copy()
        if not blocks.empty:
            all_blocks.append(calculate_distance(blocks['x'], blocks['y']))
            
        # Unblocked (PBP - Enriched)
        unblocked = df[df['event'].isin(['shot', 'goal', 'missed-shot'])].copy()
        if not unblocked.empty:
            unblocked = fit_xgs.enrich_data_with_bios(unblocked)
            unblocked['dist'] = calculate_distance(unblocked['x'], unblocked['y'])
            all_origins_f.append(unblocked[unblocked['shooter_role'] == 'F']['dist'])
            all_origins_d.append(unblocked[unblocked['shooter_role'] == 'D']['dist'])
            
    d_blocks = np.concatenate(all_blocks) if all_blocks else np.array([])
    d_origins_f = np.concatenate(all_origins_f) if all_origins_f else np.array([])
    d_origins_d = np.concatenate(all_origins_d) if all_origins_d else np.array([])
    
    print(f"Stats:\n  Blocks: {len(d_blocks)}\n  F-Origins: {len(d_origins_f)}\n  D-Origins: {len(d_origins_d)}")

    cdf_blk_global, _ = get_cdf(d_blocks)
    mappings = {
        'F': {'cdf_block': cdf_blk_global, 'icdf_origin': get_cdf(d_origins_f)[1]},
        'D': {'cdf_block': cdf_blk_global, 'icdf_origin': get_cdf(d_origins_d)[1]},
        'global': {'cdf_block': cdf_blk_global, 'icdf_origin': get_cdf(np.concatenate([d_origins_f, d_origins_d]))[1]}
    }
    
    out_path = 'puck/data/cdf_mappings.joblib'
    joblib.dump(mappings, out_path)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
