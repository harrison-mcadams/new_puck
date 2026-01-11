
import joblib
import os
import numpy as np

def investigate_defenders():
    path = os.path.join('puck', 'data', 'quantile_mappings.joblib')
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    print(f"Loading {path}...")
    mappings = joblib.load(path)
    
    role = 'D'
    if role not in mappings:
        print(f"Role {role} not found in mappings!")
        return
        
    m = mappings[role]
    
    # Normalized X: 0 (Center) to 89 (Net)
    # Low X = Far from Net (Point)
    # High X = Close to Net (Crease)
    
    ox = m['origin_x_dist']
    bx = m['block_x_dist']
    
    print(f"\n--- Role: {role} (Defender) ---")
    print("Percentiles | Block X (Loc) | Origin X (Shot) | Shift")
    
    for p in [0, 10, 25, 50, 75, 90, 100]:
        diff = ox[p] - bx[p]
        print(f"{p:3d}%       | {bx[p]:5.1f}       | {ox[p]:5.1f}           | {diff:5.1f}")
        
    print("\nInterpretation:")
    print(f"Median Defender Shot Origin X: {ox[50]:.1f}")
    if ox[50] > 60:
        print("WARNING: Median Defender Shot is closer than faceoff circles (>60). Suspect data quality.")
    elif ox[50] < 40:
         print("OK: Median Defender Shot is near blue line (<40).")
    else:
         print("Ambiguous: Median Defender Shot is in high slot (40-60).")

if __name__ == "__main__":
    investigate_defenders()
