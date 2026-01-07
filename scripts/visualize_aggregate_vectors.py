
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob

# Add project root to path
sys.path.append(os.path.join(os.getcwd()))
from puck import rink, config

def visualize_aggregate():
    summary_path = os.path.join("analysis", "blocked_shots", "blocked_shots_summary_batch.csv")
    if not os.path.exists(summary_path):
        print(f"Summary file not found: {summary_path}")
        return

    print(f"Loading summary from {summary_path}...")
    df_sum = pd.read_csv(summary_path)
    
    # Filter Score > 0.4
    df_filtered = df_sum[df_sum['score'] > 0.4]
    print(f"Found {len(df_filtered)} / {len(df_sum)} shots with score > 0.4")
    
    if df_filtered.empty:
        print("No matches found.")
        return

    # Collect Vectors
    vectors = []
    
    for idx, row in df_filtered.iterrows():
        csv_path = row['csv_path']
        if not os.path.exists(csv_path):
            continue
            
        try:
            df_cand = pd.read_csv(csv_path)
            if df_cand.empty: continue
            
            # Get the best candidate (assuming first row is best, or match frame_idx)
            best_frame = row['best_frame']
            match = df_cand[df_cand['frame_idx'] == best_frame]
            
            if not match.empty:
                cand = match.iloc[0]
                
                # Start
                x1 = cand['x']
                y1 = cand['y']
                
                # Travel Dist (calculated in identify script)
                dist = cand.get('travel_dist', 0)
                
                # Direction
                angle_rad = np.radians(cand['angle']) if 'angle' in cand else np.arctan2(cand['vy'], cand['vx'])
                
                # End (Block Location)
                # Note: 'travel_dist' in identify script is Speed * TimeToBlock
                # So End = Start + UnitVec * Dist
                # But we have vx, vy.
                # Unit Vec:
                speed = np.sqrt(cand['vx']**2 + cand['vy']**2)
                if speed > 0:
                    ux = cand['vx'] / speed
                    uy = cand['vy'] / speed
                    x2 = x1 + ux * dist
                    y2 = y1 + uy * dist
                    
                    vectors.append({
                        'x1': x1, 'y1': y1,
                        'x2': x2, 'y2': y2,
                        'score': row['score']
                    })
        except Exception as e:
            # print(f"Error reading {csv_path}: {e}")
            continue
            
    print(f"Extracted {len(vectors)} vectors for plotting.")
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    rink.draw_rink(ax)
    
    # Normalize Coords Check
    # The summary data and csv data might have varying normalization depending on when it was run?
    # No, we fixed normalization in verify script.
    # But let's assume standard rink coords (-100, 100).
    # If we see huge values, we handle them.
    
    for v in vectors:
        # Check domain
        if abs(v['x1']) > 150: # Likely raw
             v['x1'] = (v['x1'] - 1200.0) / 12.0
             v['y1'] = -(v['y1'] - 510.0) / 12.0
             v['x2'] = (v['x2'] - 1200.0) / 12.0
             v['y2'] = -(v['y2'] - 510.0) / 12.0
             
        alpha = min(0.1 + (v['score'] - 0.4), 1.0) # Higher score = more opaque
        ax.plot([v['x1'], v['x2']], [v['y1'], v['y2']], color='red', alpha=alpha, linewidth=1)
        # origin dot
        ax.scatter(v['x1'], v['y1'], s=10, c='green', alpha=alpha, edgecolors='none')
        # block x
        ax.scatter(v['x2'], v['y2'], s=10, c='black', alpha=alpha, marker='x')

    ax.set_title(f"Identified Blocked Shots (Score > 0.4)\nN={len(vectors)}")
    
    out_path = os.path.join("analysis", "blocked_shots", "aggregate_vectors_verified.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    visualize_aggregate()
