
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../puck/data/blocked_shot_model.json')

def analyze():
    if not os.path.exists(MODEL_PATH):
        print("Model not found.")
        return

    with open(MODEL_PATH, 'r') as f:
        data = json.load(f)

    bins = data.get('bins', {})
    meta = data.get('meta', {})
    bin_size = meta.get('bin_size', 5.0)
    y_min = meta.get('y_min', -50.0)

    print(f"Total Bins: {len(bins)}")
    
    # Metrics
    ns = []
    backwards_count = 0
    total_count = 0
    
    # Net Location (Attacking Right -> Net at X=89)
    NET_X = 89.0
    NET_Y = 0.0
    
    backwards_vectors = []

    for key, b_data in bins.items():
        n = b_data['n']
        ns.append(n)
        
        mx = b_data['mx']
        my = b_data['my']
        
        # Reconstruct Bin Center
        k_x, k_y = map(int, key.split('_'))
        bx = k_x * bin_size + bin_size/2
        by = k_y * bin_size + y_min + bin_size/2
        
        # Distances to Net
        dist_block = np.hypot(bx - NET_X, by - NET_Y)
        dist_origin = np.hypot(mx - NET_X, my - NET_Y)
        
        # Check if "Backwards"
        # If Origin is Closer to Net than Block, that's suspicious for a blocked shot.
        # (Allowing small margin for noise/binning measurement error)
        if dist_origin < (dist_block - 2.0): # 2ft buffer
            backwards_count += 1
            backwards_vectors.append({
                'key': key,
                'bx': bx, 'by': by,
                'ox': mx, 'oy': my,
                'n': n,
                'diff': dist_block - dist_origin
            })
            
    # Sparsity Analysis
    ns = np.array(ns)
    print(f"\n--- Sparsity ---")
    print(f"N=1: {np.sum(ns == 1)} bins ({np.sum(ns == 1)/len(bins):.1%})")
    print(f"N<3: {np.sum(ns < 3)} bins ({np.sum(ns < 3)/len(bins):.1%})")
    print(f"Max N: {np.max(ns)}")
    print(f"Mean N: {np.mean(ns):.2f}")
    
    # Validity Analysis
    print(f"\n--- Validity ---")
    print(f"Total Bins: {len(bins)}")
    print(f"Suspicious 'Backwards' Bins: {backwards_count} ({backwards_count/len(bins):.1%})")
    
    if backwards_vectors:
        print("\nTop 5 Most 'Backwards' Vectors (Origin much closer to Net):")
        backwards_vectors.sort(key=lambda x: x['diff'], reverse=True)
        for v in backwards_vectors[:5]:
            print(f"  Bin {v['key']} (N={v['n']}): Block({v['bx']:.1f}, {v['by']:.1f}) -> Origin({v['ox']:.1f}, {v['oy']:.1f}) | Closer by {v['diff']:.1f}ft")

if __name__ == "__main__":
    analyze()
