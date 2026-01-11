
"""
Script to diagnose the trained Blocked Shot Model (Defensemen).
Checks if the predicted origins for Slot Blocks are suspiciously close to the block itself.
"""
import json
import os
import math

MODEL_PATH = "puck/data/blocked_shot_model_D.json"

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        return

    with open(MODEL_PATH, 'r') as f:
        data = json.load(f)

    bins = data.get('bins', {})
    meta = data.get('meta', {})
    bin_size = meta.get('bin_size', 5.0)

    print(f"Loaded Defensemen Model with {len(bins)} bins.")
    print(f"Bin Size: {bin_size}")

    # Inspect Slot Area: X in [70, 85], Y in [-10, 10] (Attacking Zone +89)
    # Note: Model trained on 0..100 grid where X=0? No, check train script.
    # Train script: x_bins 0..100.
    # Attacking zone is usually X > 25. Net at 89.
    # So Slot is around X=80..60?
    
    # Let's inspect a few key bins in the High Slot / Hashmarks
    # Grid coordinates are i, j indices.
    
    # X=75 (20ft from net approx), Y=0 (Center)
    # i = 75 // 5 = 15
    # j = (0 + 50) // 5 = 10 (since y is -50..50)
    
    inspect_bins = [
        (15, 10), # (75, 0)
        (14, 10), # (70, 0)
        (16, 10), # (80, 0)
        (15, 11), # (75, 5)
        (15, 9),  # (75, -5)
    ]
    
    print("\n--- Inspecting Prediction for D-Block in Slot ---")
    print(f"{'Bin (i,j)':<12} | {'Block Center':<15} | {'Pred. Origin (mx, my)':<25} | {'Dist to Net':<12} | {'Shot Dist':<10} | {'N':<5} | {'Damping'}")
    
    for i, j in inspect_bins:
        k = f"{i}_{j}"
        
        # Reconstruct Block Center
        bx = i * bin_size + bin_size/2
        by = j * bin_size - 50.0 + bin_size/2
        
        if k in bins:
            b = bins[k]
            mx = b['mx']
            my = b['my']
            n = b['n']
            damp = b['damping']
            
            # Distance from Net (89, 0)
            dist_origin = math.hypot(mx - 89, my)
            
            # Shot Distance (Block to Origin)
            shot_len = math.hypot(mx - bx, my - by)
            
            print(f"({i},{j})      | ({bx:<4.1f}, {by:<4.1f})   | ({mx:<6.1f}, {my:<6.1f})          | {dist_origin:<12.1f} | {shot_len:<10.1f} | {n:<5} | {damp:.2f}")
        else:
            print(f"({i},{j})      | ({bx:<4.1f}, {by:<4.1f})   | NO DATA")

if __name__ == "__main__":
    main()
