
import sys
import os
import numpy as np
import glob
import json

# Add project root
sys.path.append(os.getcwd())
from puck import config

def check_calibration(season='20252026', condition='5v5'):
    cache_dir = os.path.join('data/cache', season, 'partials')
    pattern = os.path.join(cache_dir, f"*_{condition}.npz")
    files = glob.glob(pattern)
    
    print(f"Checking calibration for {season} {condition}")
    print(f"Found {len(files)} cache files in {cache_dir}")
    
    total_xg = 0.0
    total_goals = 0.0
    total_blocked = 0.0
    total_missed = 0.0
    total_shots = 0.0
    total_seconds = 0.0
    
    valid_files = 0
    
    for f in files:
        try:
            with np.load(f, allow_pickle=True) as data:
                if 'empty' in data:
                    continue
                
                # Iterate all team stats in this file
                # Keys: team_{tid}_stats
                keys = list(data.keys())
                for k in keys:
                    if k.startswith('team_') and k.endswith('_stats'):
                        stats_raw = data[k]
                        if stats_raw.dtype.kind in {'U', 'S'}:
                            s = json.loads(str(stats_raw.item()))
                        else:
                            s = json.loads(str(stats_raw))
                            
                        # 'team_xgs' is for the team in question
                        # We sum up for the whole league. 
                        # Note: each game has 2 teams. We will sum both sides.
                        # This gives League Total xG and League Total Goals.
                        
                        total_xg += s.get('team_xgs', 0.0)
                        total_goals += s.get('team_goals', 0)
                        total_seconds += s.get('team_seconds', 0.0)
                        
                        # attempts = shots + missed + blocked?
                        # s['team_attempts'] usually corsi
                        
                valid_files += 1
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    print("-" * 40)
    print(f"Processed {valid_files} valid cache files.")
    print(f"Total League xG:    {total_xg:.2f}")
    print(f"Total League Goals: {total_goals:.2f}")
    if total_goals > 0:
        print(f"Ratio (xG/Goals):   {total_xg / total_goals:.4f}")
    else:
        print(f"Ratio (xG/Goals):   N/A (0 goals)")
    print(f"Total Seconds:      {total_seconds:.1f} ({total_seconds/60:.1f} min)")
    print("-" * 40)
    
    if total_xg > total_goals * 1.5:
        print("FAIL: xG seems INFLATED (> 1.5x goals)")
    elif total_xg < total_goals * 0.5:
        print("FAIL: xG seems DEFLATED (< 0.5x goals)")
    else:
        print("PASS: xG is reasonably calibrated (0.5x - 1.5x goals)")

if __name__ == "__main__":
    check_calibration()
