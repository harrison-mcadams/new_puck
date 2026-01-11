
import json
import os
import pandas as pd
import numpy as np

def analyze_model(role):
    path = f'puck/data/blocked_shot_model_{role}.json'
    if not os.path.exists(path):
        print(f"Model not found: {path}")
        return

    with open(path, 'r') as f:
        data = json.load(f)

    bins = data['bins']
    meta = data['meta']
    bin_size = meta['bin_size'] # 5.0

    # Collect stats per X-bin (longitudinal)
    # Key format: "x_y" (indices)
    
    stats = []
    
    for k, v in bins.items():
        xi, yi = map(int, k.split('_'))
        
        # Convert bin index to coordinate (Center of bin)
        bx = xi * bin_size + bin_size/2
        by = yi * bin_size - 50.0 + bin_size/2 # Rough y offset from training script
        
        # We care mostly about X (distance from center/net)
        # Note: In training, X range was 0..100.
        # Blue Line is ~25. (Net at 89).
        
        stats.append({
            'x_bin_idx': xi,
            'bx_center': bx,
            'by_center': by,
            'imputed_x': v['mx'],
            'imputed_y': v['my'],
            'n': v['n'],          # Raw count in this specific bin
            'w_n': v['w_n'],      # Weighted count comparison
            'damping': v['damping']
        })

    df = pd.DataFrame(stats)
    
    if df.empty:
        print(f"No data in {role} model.")
        return

    print(f"\n--- Analysis for {role} Model ---")
    
    # Group by X-Bin to see longitudinal distribution
    # bin 0 -> x=2.5 (Center Ice / Neutral Zone?)
    # bin 5 -> x=27.5 (Blue Line)
    # bin 17 -> x=87.5 (Net)
    
    x_stats = df.groupby('x_bin_idx').agg({
        'n': 'sum',           # Total raw samples at this X depth
        'damping': 'mean',    # Average confidence/damping
        'bx_center': 'mean',  # Should be const
        'imputed_x': 'mean'   # Avg imputed origin X
    }).sort_index()
    
    # Calculate "Kickback" (Distance shot originated behind block)
    # Origin X should be < Block X (further from net, if X increases towards net)
    # Wait, check training coordinate system:
    # "Attacking Zone is +X". Net at 89. Blue line at 25.
    # So "Further from net" means Lower X.
    # Kickback = Block X - Imputed Origin X. (Positive means origin is further back).
    
    x_stats['avg_kickback'] = x_stats['bx_center'] - x_stats['imputed_x']
    
    print(f"{'X (ft)':<10} {'N Samples':<10} {'Avg Damp':<10} {'Avg Kickback':<15} {'Description'}")
    print("-" * 65)
    
    for idx, row in x_stats.iterrows():
        x_val = row['bx_center']
        n_samp = int(row['n'])
        damp = row['damping']
        kick = row['avg_kickback']
        
        desc = ""
        if x_val < 25: desc = "Neutral Zone / Far"
        elif x_val < 30: desc = "Blue Line"
        elif x_val > 80: desc = "Near Net"
        
        print(f"{x_val:<10.1f} {n_samp:<10} {damp:<10.2f} {kick:<15.1f} {desc}")

    # Check for specific "Very Far" sparsity
    far_data = df[df['bx_center'] < 25]
    print(f"\nTotal Neutral Zone (X < 25) Samples: {far_data['n'].sum()}")

if __name__ == "__main__":
    analyze_model('F')
    analyze_model('D')
