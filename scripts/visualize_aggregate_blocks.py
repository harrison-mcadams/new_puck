
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())
from puck import rink

def main():
    summary_path = "analysis/blocked_shots/blocked_shots_summary_batch.csv"
    if not os.path.exists(summary_path):
        print("Summary file not found.")
        return

    df = pd.read_csv(summary_path)
    
    # Filter Score > 0.4
    df_high = df[df['score'] > 0.4]
    print(f"Found {len(df_high)} high-score blocked shots.")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    rink.draw_rink(ax)
    
    count = 0
    for _, row in df_high.iterrows():
        csv_path = row['csv_path']
        if not os.path.exists(csv_path): continue
        
        try:
            df_cand = pd.read_csv(csv_path)
            if df_cand.empty: continue
            
            # Find best frame
            best_frame = row['best_frame']
            # Match frame in candidates
            cand = df_cand[df_cand['frame_idx'] == best_frame]
            if cand.empty: 
                # Fallback to first row (highest score)
                cand = df_cand.iloc[0]
            else:
                cand = cand.iloc[0]
                
            # Get Vector
            x = cand['x']
            y = cand['y']
            vx = cand['vx']
            vy = cand['vy']
            
            # Check Norm
            if abs(x) > 200:
                x = (x - 1200.0) / 12.0
                y = -(y - 510.0) / 12.0
            
            # Vector Scale
            # Use 'travel_dist' (Shot Origin to Impact) if available.
            # Fallback to 'blocker_dist' (Proximity) which is often small, so be careful.
            travel_dist = cand.get('travel_dist', 0.0)
            blocker_dist = cand.get('blocker_dist', 0.0)
            
            length = 20.0 # Default
            if pd.notnull(travel_dist) and travel_dist > 2.0:
                length = travel_dist
            elif pd.notnull(blocker_dist) and blocker_dist > 5.0:
                length = blocker_dist
            
            spd_mag = np.sqrt(vx**2 + vy**2)
            if spd_mag > 0.001:
                dx = (vx / spd_mag) * length
                dy = (vy / spd_mag) * length
            else:
                dx = vx
                dy = vy
            
            # Plot
            ax.arrow(x, y, dx, dy, head_width=1, head_length=2, fc='red', ec='red', alpha=0.3, width=0.2)
            count += 1
            
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")
            continue

    ax.set_title(f"Aggregate Blocked Shots (Score > 0.4) - {count} Vectors", fontsize=16)
    
    out_path = "analysis/aggregate_blocked_shots.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved aggregate plot to {out_path}")

if __name__ == "__main__":
    main()
