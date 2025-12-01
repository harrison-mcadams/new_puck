
import analyze
import os
import numpy as np
import json

def debug_baseline():
    season = '20252026'
    teams = ['ANA', 'BOS']
    conditions = {
        '5v5': {'game_state': ['5v5'], 'is_net_empty': [0]}
    }
    out_dir = f'static/league/{season}'
    
    print(f"Running debug analysis for {teams}...")
    results = analyze.season(
        season=season,
        conditions=conditions,
        baseline_mode='compute',
        teams=teams,
        out_dir=out_dir
    )
    
    # Check baseline
    out_dir = f'static/league/{season}'
    baseline_path = os.path.join(out_dir, '5v5', f'{season}_league_baseline.npy')
    json_path = os.path.join(out_dir, '5v5', f'{season}_league_baseline.json')
    
    if os.path.exists(baseline_path):
        baseline = np.load(baseline_path)
        print(f"Baseline shape: {baseline.shape}")
        print(f"Baseline sum: {np.nansum(baseline)}")
        print(f"Baseline max: {np.nanmax(baseline)}")
        if np.nansum(baseline) == 0:
            print("FAILURE: Baseline is empty (sum is 0).")
        else:
            print("SUCCESS: Baseline has data.")
    else:
        print("FAILURE: Baseline NPY not found.")
        
    if os.path.exists(json_path):
        print(f"Reading from {os.path.abspath(json_path)}")
        with open(json_path, 'r') as f:
            data = json.load(f)
        print("Baseline JSON:", json.dumps(data, indent=2))
    else:
        print("FAILURE: Baseline JSON not found.")

if __name__ == "__main__":
    debug_baseline()
