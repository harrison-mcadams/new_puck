
import joblib
import os
import numpy as np

def debug_mappings():
    path = os.path.join('puck', 'data', 'quantile_mappings.joblib')
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    print(f"Loading {path}...")
    mappings = joblib.load(path)
    
    print("Keys found:", mappings.keys())
    
    for role in mappings:
        print(f"\n--- Role: {role} ---")
        m = mappings[role]
        print("Keys:", m.keys())
        
        if 'block_x_dist' in m:
            bx = m['block_x_dist']
            ox = m['origin_x_dist']
            print(f"Block X Percentiles (0, 50, 100): {bx[0]:.2f}, {bx[50]:.2f}, {bx[-1]:.2f}")
            print(f"Origin X Percentiles (0, 50, 100): {ox[0]:.2f}, {ox[50]:.2f}, {ox[-1]:.2f}")
            
            # Analyze Shift Magnitude
            # shift = Origin - Block
            # (Note: normalized coords, so positive/negative meanings depend on coordinate system)
            # If Net=89, Block=80, Origin=40. Shift = 40-80 = -40 (Large negative shift away from net)
            
            diffs = ox - bx
            print(f"Mean Shift (Origin - Block): {np.mean(diffs):.2f}")
            print(f"Median Shift: {np.median(diffs):.2f}")
            print(f"Shift at 10th percentile (Deep Block): {ox[10] - bx[10]:.2f}")
            print(f"Shift at 90th percentile (High Block): {ox[90] - bx[90]:.2f}")
            
            print("--- Table (Percentile | Block X -> Origin X | Shift) ---")
            for p in [0, 10, 25, 50, 75, 90, 100]:
                 print(f"{p:3d}% | {bx[p]:5.1f} -> {ox[p]:5.1f} | {ox[p]-bx[p]:5.1f}")

if __name__ == "__main__":
    debug_mappings()
