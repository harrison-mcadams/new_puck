import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import glob
import numpy as np
import sys
sys.path.append(os.getcwd())
from puck import rink

def visualize_block(match_index=0):
    # 1. Load Discovery Results
    df_results = pd.read_csv('analysis/edge_block_discovery.csv')
    if df_results.empty:
        print("No matches found in analysis/edge_block_discovery.csv")
        return

    df_results['game_id'] = df_results['game_id'].astype(int)
    df_results['goal_event_id'] = df_results['goal_event_id'].astype(int)

    # Select a specific block to visualize
    # Let's try to find Game 2024020202 if available, else first
    target_game = 2024020202
    target_event = 328
    
    target_rows = df_results[
        (df_results['game_id'] == target_game) &
        (df_results['goal_event_id'] == target_event)
    ]
    
    if not target_rows.empty:
        row = target_rows.iloc[0]
        print(f"Visualizing Target Game {target_game}...")
    else:
        row = df_results.iloc[0]
        print(f"Target game not found. Visualizing first available: Game {row['game_id']}...")
    
    game_id = int(row['game_id'])
    goal_event_id = int(row['goal_event_id'])
    
    print(f"Selected Game: {game_id}, Goal: {goal_event_id}")
    print(f"PBP Block: ({row['pbp_x']:.1f}, {row['pbp_y']:.1f})")
    print(f"True Block: ({row['true_block_x']:.1f}, {row['true_block_y']:.1f})")
    print(f"True Origin: ({row['true_origin_x']:.1f}, {row['true_origin_y']:.1f})")
    
    # 2. Get Game Link
    print(f"Game Link: https://www.nhl.com/gamecenter/{game_id}")
    
    # 3. Load Tracking Data
    season = str(game_id)[:4] + str(int(str(game_id)[:4]) + 1) # e.g. 20242025
    
    # Try finding file
    pattern = os.path.join('data', 'edge_goals', season, f"game_{game_id}_goal_{goal_event_id}_*.csv")
    files = glob.glob(pattern)
    if not files:
        print(f"Tracking file not found: {pattern}")
        return
        
    pos_csv = files[0]
    df_pos = pd.read_csv(pos_csv)
    
    # Filter for puck
    df_puck_raw = df_pos[df_pos['entity_type'] == 'puck'].copy()
    
    # Sort and calculate kinematics
    df_puck_raw = df_puck_raw.sort_values('frame_idx')
    
    # Calculate Velocity and Angle for visualization
    df_puck_raw['dx'] = df_puck_raw['x'].diff()
    df_puck_raw['dy'] = df_puck_raw['y'].diff()
    df_puck_raw['dt'] = df_puck_raw['timestamp'].diff() / 10.0  # Deciseconds to Seconds
    # Handle first row NaNs
    df_puck_raw['dx'] = df_puck_raw['dx'].fillna(0)
    df_puck_raw['dy'] = df_puck_raw['dy'].fillna(0)
    df_puck_raw['dt'] = df_puck_raw['dt'].replace(0, np.nan).fillna(0.1) # Avoid div/0
    
    df_puck_raw['vx'] = df_puck_raw['dx'] / df_puck_raw['dt']
    df_puck_raw['vy'] = df_puck_raw['dy'] / df_puck_raw['dt']
    df_puck_raw['speed'] = np.sqrt(df_puck_raw['vx']**2 + df_puck_raw['vy']**2)
    
    # Normalize Coordinates if needed (check range)
    # Script discovered blocks logic: if max > 200, normalize.
    is_raw = df_puck_raw['x'].abs().max() > 200
    if is_raw:
        df_puck_raw['x'] = (df_puck_raw['x'] - 1200.0) / 12.0
        df_puck_raw['y'] = -(df_puck_raw['y'] - 510.0) / 12.0
    
    df_puck = df_puck_raw.sort_values('frame_idx')
    
    # Trim to relevant window (Origin to Goal)
    # Origin logic in discovery script was: find release
    # We can just finding the closest frames to the recorded True Origin and True Block
    
    # Find frame for True Origin
    # dist = sqrt((x-ox)^2 + (y-oy)^2)
    df_puck['dist_origin'] = np.sqrt((df_puck['x'] - row['true_origin_x'])**2 + (df_puck['y'] - row['true_origin_y'])**2)
    origin_frame_idx = df_puck.loc[df_puck['dist_origin'].idxmin(), 'frame_idx']
    
    # Sort by dist to Origin
    df_puck['dist_to_origin'] = np.sqrt(
        (df_puck['x'] - row['true_origin_x'])**2 + 
        (df_puck['y'] - row['true_origin_y'])**2
    )
    # Start: Close to Origin
    origin_frame_idx = df_puck.loc[df_puck['dist_to_origin'].idxmin()]['frame_idx']
    
    # End: Goal Frame + Buffer
    goal_frame_idx = int(row['goal_frame'])
    block_frame_idx = df_puck.loc[(df_puck['x'] - row['true_block_x']).abs().idxmin()]['frame_idx']
    
    # Ensure our window covers Origin -> Block -> Goal -> Post-Goal Buffer
    start_frame = max(0, origin_frame_idx - 10)
    end_frame = goal_frame_idx + 20  # +2.0 seconds after goal to see it cross line
    
    df_clip = df_puck[
        (df_puck['frame_idx'] >= start_frame) & 
        (df_puck['frame_idx'] <= end_frame)
    ].copy()
    # --- PLOT 1: STILLS ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Still 1: True Origin
    ax = axes[0]
    rink.draw_rink(ax)
    ax.plot(df_clip['x'], df_clip['y'], 'k-', alpha=0.3)
    ax.plot(row['true_origin_x'], row['true_origin_y'], 'g*', markersize=15, label='True Origin (Release)')
    ax.set_title(f"True Origin (Release)\n({row['true_origin_x']:.1f}, {row['true_origin_y']:.1f})")
    ax.legend()
    
    # Still 2: Block Comparison
    ax = axes[1]
    rink.draw_rink(ax)
    # Plot the full available tracking data for context
    # Use 'cool' colormap for time/speed
    scatter = ax.scatter(df_puck['x'], df_puck['y'], c=df_puck['frame_idx'], cmap='viridis', s=10, alpha=0.5, label='Full Path')
    
    # Add arrows to show directionality (every 5th frame)
    quiver_df = df_puck.iloc[::5]
    ax.quiver(quiver_df['x'], quiver_df['y'], quiver_df['dx'], quiver_df['dy'], 
              color='gray', alpha=0.5, width=0.003, headwidth=3, scale=None)

    # Highlight the specific "Match" points
    ax.scatter(row['true_origin_x'], row['true_origin_y'], c='green', s=200, marker='*', label='Detected Origin', zorder=10)
    ax.scatter(row['true_block_x'], row['true_block_y'], c='red', s=150, marker='x', label='Detected Deflection', zorder=10)
    
    # --- PBP Locations (Pre-Calculated) ---
    # Original
    ax.scatter(row['pbp_x'], row['pbp_y'], c='blue', s=80, alpha=0.3, label='PBP (Original)', zorder=9)
    
    # Aligned (Cyan Dot)
    # Check if we have the aligned columns
    if 'pbp_aligned_x' in row:
         ax.scatter(row['pbp_aligned_x'], row['pbp_aligned_y'], c='cyan', s=120,    marker='o', edgecolors='black', linewidth=2, label='PBP (Aligned)', zorder=11)
    
    # --- Timing Context ---
    # Highlight the frame where PBP says the block happened
    if 'pbp_tracking_frame' in row and not pd.isna(row['pbp_tracking_frame']):
        pbp_frame = int(row['pbp_tracking_frame'])
        # Find puck position at this frame
        puck_at_frame = df_puck[df_puck['frame_idx'] == pbp_frame]
        if not puck_at_frame.empty:
            px = puck_at_frame.iloc[0]['x']
            py = puck_at_frame.iloc[0]['y']
            ax.scatter(px, py, c='cyan', s=70, marker='s', label='Puck @ PBP Time', zorder=12)

    # --- Goal Frame Marker ---
    # Show exactly where the logic thinks the goal happened
    goal_frame = int(row['goal_frame'])
    puck_at_goal = df_puck[df_puck['frame_idx'] == goal_frame]
    if not puck_at_goal.empty:
        gx = puck_at_goal.iloc[0]['x']
        gy = puck_at_goal.iloc[0]['y']
        ax.scatter(gx, gy, c='magenta', s=150, marker='H', edgecolors='black', label='Detected Goal', zorder=15)
        
        # Annotate Goal
        ax.text(gx, gy + 3, f"GOAL Frame {goal_frame}", color='magenta', fontweight='bold', ha='center')

    ax.set_title(f"Game {game_id} Event {goal_event_id}\n(Green=Origin, Red=Deflection, Cyan=PBP Time, Magenta=Goal)")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    
    plt.tight_layout()
    os.makedirs('analysis/plots', exist_ok=True)
    plt.savefig('analysis/plots/block_verification_stills.png')
    print("Saved stills to analysis/plots/block_verification_stills.png")
    plt.close()

    # --- PLOT 2: ANIMATION ---
    fig, ax = plt.subplots(figsize=(10, 6))
    rink.draw_rink(ax)
    
    line, = ax.plot([], [], 'k-', linewidth=2)
    puck_point, = ax.plot([], [], 'ko', markersize=8)
    
    # Fixed markers
    origin_mark, = ax.plot([], [], 'g*', markersize=15, label='True Origin')
    block_mark, = ax.plot([], [], 'rx', markersize=15, markeredgewidth=3, label='True Block')
    goal_mark, = ax.plot([], [], 'mH', markersize=15, markeredgecolor='black', label='Detected Goal')
    pbp_mark, = ax.plot(row['pbp_x'], row['pbp_y'], 'bo', markersize=10, alpha=0.5, label='PBP Block')
    
    title = ax.text(0, 45, "", ha='center', fontsize=12)
    
    ax.legend(loc='upper right')
    
    frames = df_clip.to_dict('records')
    
    def init():
        line.set_data([], [])
        puck_point.set_data([], [])
        origin_mark.set_data([], [])
        block_mark.set_data([], [])
        goal_mark.set_data([], [])
        title.set_text("")
        return line, puck_point, origin_mark, block_mark, goal_mark, title
        
    def update(frame):
        # Current data up to frame
        curr_idx = frame['frame_idx']
        
        # Trail
        trail = df_clip[df_clip['frame_idx'] <= curr_idx]
        line.set_data(trail['x'], trail['y'])
        
        # Puck
        puck_point.set_data([frame['x']], [frame['y']])
        
        # Markers appear when passed
        if curr_idx >= origin_frame_idx:
            origin_mark.set_data([row['true_origin_x']], [row['true_origin_y']])
        else:
            origin_mark.set_data([], [])
            
        if curr_idx >= block_frame_idx:
            block_mark.set_data([row['true_block_x']], [row['true_block_y']])
        else:
            block_mark.set_data([], [])
            
        if curr_idx >= goal_frame:
            # We need coordinates for the goal frame. row['goal_frame'] is an index.
            # We can lookup the coords from df_puck if we want exact, 
            # OR just plot it near the net if that frame isn't in clip (checked earlier it is)
            # Let's find coords for goal frame
            gf_row = df_puck[df_puck['frame_idx'] == goal_frame]
            if not gf_row.empty:
                goal_mark.set_data([gf_row.iloc[0]['x']], [gf_row.iloc[0]['y']])
        else:
            goal_mark.set_data([], [])

        title.set_text(f"Frame {curr_idx} | Speed: {frame.get('speed', 0):.1f} ft/s") # Speed might not be in df_pos unless we recalc
        
        return line, puck_point, origin_mark, block_mark, goal_mark, title

    ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=50) # 50ms = 20fps (slowmo)
    
    save_path = 'analysis/plots/block_verification.gif'
    ani.save(save_path, writer='pillow')
    print(f"Saved animation to {save_path}")
    plt.close()

if __name__ == "__main__":
    visualize_block(0) # Visualize the top match by angle change
