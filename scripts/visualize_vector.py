
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def visualize_vector(game_id, goal_id, frame_idx):
    # Construct path
    search_path = os.path.join("data", "edge_goals", "20242025", f"game_{game_id}_goal_{goal_id}_positions.csv")
    if not os.path.exists(search_path):
        import glob
        pat = os.path.join("data", "edge_goals", "20242025", f"game_{game_id}_goal_{goal_id}_*.csv")
        files = glob.glob(pat)
        if files:
            search_path = files[0]
        else:
            print(f"File not found: {search_path}")
            return

    print(f"Loading {search_path}...")
    df = pd.read_csv(search_path)
    df_puck = df[df['entity_type'] == 'puck'].copy()
    df_puck = df_puck.sort_values('frame_idx').reset_index(drop=True)
    
    # Calculate Velocity
    df_puck['vx'] = df_puck['x'].diff() / 0.1
    df_puck['vy'] = df_puck['y'].diff() / 0.1
    
    # Filter Window
    window = df_puck[(df_puck['frame_idx'] >= int(frame_idx) - 10) & (df_puck['frame_idx'] <= int(frame_idx) + 10)]
    
    if window.empty:
        print("No data in window.")
        return

    # Get specific frame data
    row = df_puck[df_puck['frame_idx'] == int(frame_idx)]
    if row.empty:
        print(f"Frame {frame_idx} not found in puck data.")
        return
    row = row.iloc[0]
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.xlim(-100, 100)
    plt.ylim(-45, 45)
    
    # Draw Rink Outline (General)
    plt.axvline(-89, color='red', linestyle='--', label='Net Left')
    plt.axvline(89, color='red', linestyle='--', label='Net Right')
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    
    # Plot Trajectory
    plt.plot(window['x'], window['y'], 'o-', markersize=4, label='Puck Path')
    
    # Plot Vector at Frame
    # Arrow vector direction
    plt.arrow(row['x'], row['y'], row['vx']*0.2, row['vy']*0.2, 
              head_width=2, head_length=3, fc='k', ec='k', label='Vector (F20)')
              
    plt.title(f"Game {game_id} Goal {goal_id} - Vector F{frame_idx}\nPuck ({row['x']:.1f}, {row['y']:.1f}) -> V({row['vx']:.1f}, {row['vy']:.1f})")
    plt.xlabel("X (feet)")
    plt.ylabel("Y (feet)")
    plt.legend()
    plt.grid(True)
    
    # Save to ARTIFACTS directory
    # But wait, python script runs in CWD.
    # User can see "c:\Users\harri\Desktop\new_puck\vector_viz.png".
    out_file = f"vector_viz_{game_id}_{goal_id}_f{frame_idx}.png"
    plt.savefig(out_file)
    print(f"Saved visualization to {os.path.abspath(out_file)}")

if __name__ == "__main__":
    visualize_vector(sys.argv[1], sys.argv[2], sys.argv[3])
