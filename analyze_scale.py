import numpy as np
import os
import glob

def analyze_distribution():
    base_path = 'static/league/20252026/5v5'
    int_dir = os.path.join(base_path, 'intermediates')
    baseline_path = os.path.join(base_path, '20252026_league_baseline.npy')
    
    if not os.path.exists(baseline_path):
        print("Baseline not found.")
        return

    league_baseline = np.load(baseline_path)
    
    files = glob.glob(os.path.join(int_dir, '*_maps.npz'))
    print(f"Found {len(files)} intermediate files.")
    
    all_diffs = []
    
    for f in files:
        try:
            data = np.load(f, allow_pickle=True)
            team_map = data['team_map']
            stats = data['stats'].item()
            team_seconds = stats.get('team_seconds', 1.0)
            
            # Normalize
            team_map_norm = team_map / team_seconds * 3600.0
            
            # Diff
            diff = team_map_norm - league_baseline
            all_diffs.append(diff.flatten())
        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    if not all_diffs:
        print("No data found.")
        return
        
    combined = np.concatenate(all_diffs)
    combined = combined[~np.isnan(combined)]
    abs_combined = np.abs(combined)
    
    print(f"Total valid pixels: {combined.size}")
    print(f"Max: {combined.max()}")
    
    # Percentiles for linthresh tuning
    # We want linthresh to be around the "noise floor" or median non-zero value?
    # Or just enough to show the low-density structure.
    for p in [50, 60, 70, 80, 90, 95, 99]:
        val = np.percentile(abs_combined, p)
        print(f"{p}th percentile of abs diff: {val}")

analyze_distribution()
