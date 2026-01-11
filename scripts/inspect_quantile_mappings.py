
import joblib
import os
import sys
import numpy as np

def main():
    # Path to joblib file (relative to script or project root)
    # Assumes running from project root
    path = "puck/data/quantile_mappings.joblib"
    
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return

    print(f"Loading {path}...")
    mappings = joblib.load(path)
    
    print(f"Keys found: {mappings.keys()}")
    
    for role in ['global', 'F', 'D']:
        if role not in mappings:
            print(f"\n--- Role: {role} (NOT FOUND) ---")
            continue
            
        print(f"\n--- Role: {role} ---")
        m = mappings[role]
        
        # Check keys
        print(f"  Mapping Keys: {m.keys()}")
        
        if 'block_dist_net' in m and 'origin_dist_net' in m:
            bd = m['block_dist_net']
            od = m['origin_dist_net']
            
            print(f"  Start (0%): Block {bd[0]:.1f} -> Origin {od[0]:.1f}")
            print(f"  25%:        Block {bd[25]:.1f} -> Origin {od[25]:.1f}")
            print(f"  50% (Med):  Block {bd[50]:.1f} -> Origin {od[50]:.1f}")
            print(f"  75%:        Block {bd[75]:.1f} -> Origin {od[75]:.1f}")
            print(f"  End (100%): Block {bd[-1]:.1f} -> Origin {od[-1]:.1f}")
            
            # Check overlap density
            # Fraction of Origins < 20ft?
            # Since array is sorted percentiles 0..100
            # We can just count how many values are < 20
            pct_close = np.sum(od < 20)
            print(f"  % Origins < 20ft: {pct_close}%")
            
        else:
            print("  MISSING 'block_dist_net' or 'origin_dist_net'")

if __name__ == "__main__":
    main()
