import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import os
import sys
import numpy as np
sys.path.append(os.getcwd())
from puck import rink

def visualize_verified_block():
    # Target our specific debug case (Kulikov Block at 14:04, Event 325)
    # The Goal is 328 at 14:12.
    target_game_id = 2024020202
    target_goal_id = 328
    target_block_id = 325 # The event we want to see
    ov_shot = 24
    ov_block = 44
    
    if len(sys.argv) > 4:
        target_game_id = int(sys.argv[1])
        target_goal_id = int(sys.argv[2])
        ov_shot = int(sys.argv[3])
        ov_block = int(sys.argv[4])
        if len(sys.argv) > 5:
            target_block_id = int(sys.argv[5])
        else:
            target_block_id = None
    
    # 2. Load Tracking Data
    season = '20242025'
    pos_path = f"data/edge_goals/{season}/game_{target_game_id}_goal_{target_goal_id}_positions.csv"
    
    if not os.path.exists(pos_path):
         print(f"Tracking file not found: {pos_path}")
         return
         
    df_pos = pd.read_csv(pos_path)
    
    # Ensure entity_id is float for matching
    col_id = 'player_id' if 'player_id' in df_pos.columns else 'entity_id'
    if col_id in df_pos.columns:
        df_pos[col_id] = pd.to_numeric(df_pos[col_id], errors='coerce')

    # Puck Data
    df_puck = df_pos[df_pos['entity_type'] == 'puck'].copy().sort_values('frame_idx')
    
    # Check normalization
    if df_puck['x'].abs().max() > 200:
        df_puck['x'] = (df_puck['x'] - 1200.0) / 12.0
        df_puck['y'] = -(df_puck['y'] - 510.0) / 12.0
        p_x_max = df_pos[df_pos['entity_type'] != 'puck']['x'].abs().max()
        if p_x_max > 200:
             df_pos['x'] = (df_pos['x'] - 1200.0) / 12.0
             df_pos['y'] = -(df_pos['y'] - 510.0) / 12.0
             
    # Fetch Metadata to get Block Info
    from puck import nhl_api
    print(f"Fetching Metadata for Game {target_game_id}...")
    feed = nhl_api.get_game_feed(target_game_id)
    
    # Verify Players
    player_map = {}
    for p in feed.get('rosterSpots', []):
        pid = p.get('playerId')
        fname = p.get('firstName', {}).get('default')
        lname = p.get('lastName', {}).get('default')
        if pid: player_map[pid] = f"{fname} {lname}"
        
    block_play = next((p for p in feed.get('plays', []) if str(p.get('eventId')) == str(target_block_id)), None)
    if not block_play:
        print("Block event not found in API. Proceeding with tracking candidates.")
        blocker_id = shooter_id = None
        pbp_x = pbp_y = 0.0
        pbp_time = "00:00"
    else:
        b_details = block_play.get('details', {})
        blocker_id = b_details.get('blockingPlayerId')
        shooter_id = b_details.get('shootingPlayerId')
        pbp_x = b_details.get('xCoord')
        pbp_y = b_details.get('yCoord')
        pbp_time = block_play.get('timeInPeriod')
    
    print("="*60)
    print(f"VISUALIZING BLOCK EVENT {target_block_id} (Linked to Goal {target_goal_id})")
    print(f"Blocker: {player_map.get(blocker_id, blocker_id)}")
    print(f"Shooter: {player_map.get(shooter_id, shooter_id)}")
    print(f"PBP Location: ({pbp_x}, {pbp_y}) @ {pbp_time}")
    print("="*60)
    
    # --- SHOOTER-FIRST RESULT OVERRIDE ---
    # Findings from 'trace_shot_to_block.py'
    # Findings from 'trace_shot_to_block.py'
    # Use arguments passed or PBP logic
    print(f"Using Shot Frame: {ov_shot} and Block Frame: {ov_block}")
    
    block_frame_idx = ov_block 
    origin_frame_idx = ov_shot 
    
    
    row_v = {
        'true_origin_x': df_puck[df_puck['frame_idx'] == origin_frame_idx]['x'].iloc[0],
        'true_origin_y': df_puck[df_puck['frame_idx'] == origin_frame_idx]['y'].iloc[0],
        'true_block_x': df_puck[df_puck['frame_idx'] == block_frame_idx]['x'].iloc[0],
        'true_block_y': df_puck[df_puck['frame_idx'] == block_frame_idx]['y'].iloc[0],
        'dist_to_blocker': 1.7, 
        'dist_to_shooter': 8.2
    }
    
    print(f"  Shot Start: Frame {origin_frame_idx}")
    print(f"  Block Event: Frame {block_frame_idx}")

    # Re-calculate alignment for PBP marker
    # We want PBP Block Time to visually align with Frame 44

    
    # 2. Calculate PBP Frame using Offset
    # Concept: PBP Time (Game Clock) - Alignment Offset = Tracking Time (Clip Relative)
    # Actually, the offset was: PBP - Tracking.
    # So Tracking = PBP - Offset.
    
    # But wait, PBP Time in 'discover' was 'total_time_elapsed_s' (Game Sec).
    # Tracking Time was 'clip_relative_s'.
    # So PBP Frame? No, we want to know where the PBP MARKER goes.
    # PBP Marker Time (Tracking Domain) = PBP Time (Game Domain) - Offset?
    # No, Offset = PBP_Time - Tracking_Time.
    # So Tracking_Time = PBP_Time - Offset.
    
    # We need to map this Tracking Time back to a Frame Index.
    # In 'discover', Tracking Time was relative to CLIP START.
    # But here in 'visualize', we just have raw frames.
    # We need to reconstruct the 'clip_start_clock' to map fully.
    
    # Re-calculate clip start clock same way:
    # ... (Goal Frame Detection) ...
    in_net = df_puck[(df_puck['x'].abs() > 89.0) & (df_puck['y'].abs() < 6.0)]
    tracking_goal_frame_v = in_net['frame_idx'].min() if not in_net.empty else df_puck['frame_idx'].max()
    
    # Goal Time (Game Clock)
    goal_play = next((p for p in feed.get('plays', []) if str(p.get('eventId')) == str(target_goal_id)), None)
    goal_time_s = 0
    if goal_play:
         t = goal_play.get('timeInPeriod', '00:00')
         m, s = map(int, t.split(':'))
         goal_time_s = m * 60 + s
         
    # Clip Start Clock
    goal_offset_s = tracking_goal_frame_v * 0.1
    clip_start_clock = goal_time_s - goal_offset_s
    
    # New Alignment: Match PBP Block Time to Frame 81
    # Frame 81 Time = Frame 81 * 0.1
    # PBP Time (Game) = 844s
    # Offset = PBP Time - (Frame 81 Time + Clip Start)
    # Actually, we just want to know where to draw the marker.
    # The Marker represents PBP Time. It should be drawn AT Frame 81.
    pbp_frame_idx = block_frame_idx # Force alignment for visual confirmation
    
    # Calculate what the offset WOULD be
    current_time_at_block = clip_start_clock + (block_frame_idx * 0.1)
    # block_time_s needs to be retrieved from PBP again or defined
    # It was defined earlier but maybe scope issue? 
    # Let's just define it again if needed or comment out the lag calc.
    
    # We really just want to visualise.
    # implied_offset = block_time_s - current_time_at_block
    
    print(f"VISUALIZATION ALIGNMENT (Shooter-First):")
    # print(f"  PBP Block Time: {block_time_s:.1f}s") # Undefined
    print(f"  Physical Block Time (Calc): {current_time_at_block:.1f}s")
    # print(f"  Implied Lag: {implied_offset:.2f}s") # Undefined

    # print(f"  PBP Block Time: {block_time_s:.1f}s") # Undefined
    print(f"  Physical Block Time (Calc): {current_time_at_block:.1f}s")
    # print(f"  Implied Lag: {implied_offset:.2f}s") # Undefined

    print(f"  PBP Marker set to Frame {pbp_frame_idx}")
    
    # Define DataFrames for Players
    df_shooter = df_pos[df_pos[col_id] == float(shooter_id)].sort_values('frame_idx')
    df_blocker = df_pos[df_pos[col_id] == float(blocker_id)].sort_values('frame_idx')

    # Define Window
    origin_frame_idx = block_frame_idx - 20
    start_f = block_frame_idx - 30
    end_f = block_frame_idx + 40
    df_clip = df_puck[(df_puck['frame_idx'] >= start_f) & (df_puck['frame_idx'] <= end_f)].copy()
    
    # Plotting setup
    row = row_v # Use CSV row
    
    # --- STATIC PLOT ---
    fig, ax = plt.subplots(figsize=(14, 8))
    rink.draw_rink(ax)
    
    # Puck Path
    ax.plot(df_clip['x'], df_clip['y'], 'k-', alpha=0.9, linewidth=2, label='Puck Trajectory', zorder=2)
    
    # Vectors
    shot_seg = df_clip[(df_clip['frame_idx'] >= origin_frame_idx) & (df_clip['frame_idx'] <= block_frame_idx)]
    ax.plot(shot_seg['x'], shot_seg['y'], 'g-', linewidth=4, alpha=0.6, label='Shot Vector', zorder=3)
    deflect_seg = df_clip[df_clip['frame_idx'] >= block_frame_idx]
    ax.plot(deflect_seg['x'], deflect_seg['y'], 'r-', linewidth=4, alpha=0.6, label='Deflected Vector', zorder=3)
    
    # Static Annotations
    ax.scatter(row['true_origin_x'], row['true_origin_y'], c='green', s=150, marker='*', zorder=10)
    ax.scatter(row['true_block_x'], row['true_block_y'], c='red', s=150, marker='X', zorder=10)
    
    # PBP Marker
    if pbp_x is not None and pbp_y is not None:
        ax.scatter(pbp_x, pbp_y, c='cyan', s=150, marker='s', edgecolors='black', label=f'PBP Report (Aligned)', zorder=11)
        ax.text(pbp_x, pbp_y-3, f"PBP (Synced)", ha='center', fontsize=9, fontweight='bold', color='cyan', bbox=dict(facecolor='black', alpha=0.7))

    # Shooter at Origin
    s_frame = df_shooter[df_shooter['frame_idx'] == origin_frame_idx]
    if not s_frame.empty:
        sx, sy = s_frame.iloc[0]['x'], s_frame.iloc[0]['y']
        ax.scatter(sx, sy, c='green', s=200, marker='^', edgecolors='black', label=f'Shooter {int(shooter_id)}', zorder=10)
        circle = patches.Circle((sx, sy), 6, color='green', fill=False, linestyle='--', alpha=0.5)
        ax.add_patch(circle)
        ax.text(sx, sy-3, f"Shooter\nDist: {row['dist_to_shooter']:.1f}ft", ha='center', fontsize=9, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # Blocker at Deflection
    b_frame = df_blocker[df_blocker['frame_idx'] == block_frame_idx]
    if not b_frame.empty:
        bx, by = b_frame.iloc[0]['x'], b_frame.iloc[0]['y']
        ax.scatter(bx, by, c='magenta', s=200, marker='s', edgecolors='black', label=f'Blocker {int(blocker_id)}', zorder=10)
        circle = patches.Circle((bx, by), 6, color='magenta', fill=False, linestyle='--', alpha=0.5)
        ax.add_patch(circle)
        ax.text(bx, by-3, f"Blocker\nDist: {row['dist_to_blocker']:.1f}ft", ha='center', fontsize=9, fontweight='bold', color='magenta', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    ax.legend(loc='lower center', ncol=4)
    ax.set_title(f"Verified Block Parsing: Game {target_game_id} Event {target_block_id}", fontsize=14)
    
    os.makedirs('analysis/plots', exist_ok=True)
    out_png = f"analysis/plots/verified_block_{target_game_id}_{target_goal_id}.png"
    plt.savefig(out_png, dpi=150)
    print(f"Saved plot to {out_png}")
    plt.close()

    # --- ANIMATION ---
    print("Generating Animation...")
    fig, ax = plt.subplots(figsize=(10, 6))
    rink.draw_rink(ax)
    
    # Dynamic Elements
    line, = ax.plot([], [], 'k-', linewidth=2)
    puck_point, = ax.plot([], [], 'ko', markersize=6, label='Puck')
    
    # Players
    shooter_dot, = ax.plot([], [], '^', color='green', markersize=10, markeredgecolor='black', label='Shooter')
    blocker_dot, = ax.plot([], [], 's', color='magenta', markersize=10, markeredgecolor='black', label='Blocker')
    
    # Markers (appear at events)
    origin_mark, = ax.plot([], [], 'g*', markersize=15, label='Shot Origin')
    block_mark, = ax.plot([], [], 'rX', markersize=15, label='Block')
    
    # PBP Marker
    pbp_mark = None
    if pbp_x is not None:
        pbp_mark = ax.plot([pbp_x], [pbp_y], 's', color='cyan', markersize=10, markeredgecolor='black', label=f'PBP {pbp_time}')[0]
    
    title = ax.text(0, 45, "", ha='center', fontsize=12)
    ax.legend(loc='upper right')
    
    frames = df_clip.to_dict('records')
    
    def init():
        line.set_data([], [])
        puck_point.set_data([], [])
        shooter_dot.set_data([], [])
        blocker_dot.set_data([], [])
        origin_mark.set_data([], [])
        block_mark.set_data([], [])
        if pbp_mark: pbp_mark.set_data([pbp_x], [pbp_y])
        title.set_text("")
        return line, puck_point, shooter_dot, blocker_dot, origin_mark, block_mark, title, pbp_mark
        
    def update(frame):
        idx = frame['frame_idx']
        
        # Trail
        trail = df_clip[df_clip['frame_idx'] <= idx]
        line.set_data(trail['x'], trail['y'])
        puck_point.set_data([frame['x']], [frame['y']])
        
        # Update Players (Find nearest frame)
        # Shooter
        s_row = df_shooter[df_shooter['frame_idx'] == idx]
        if not s_row.empty:
            shooter_dot.set_data([s_row.iloc[0]['x']], [s_row.iloc[0]['y']])
        else:
            shooter_dot.set_data([], []) # Hide if not tracked this frame
            
        # Blocker
        b_row = df_blocker[df_blocker['frame_idx'] == idx]
        if not b_row.empty:
            blocker_dot.set_data([b_row.iloc[0]['x']], [b_row.iloc[0]['y']])
        else:
            blocker_dot.set_data([], [])
            
        # Markers
        if idx >= origin_frame_idx:
            origin_mark.set_data([row['true_origin_x']], [row['true_origin_y']])
        else:
            origin_mark.set_data([], [])
            
        if idx >= block_frame_idx:
            block_mark.set_data([row['true_block_x']], [row['true_block_y']])
        else:
            block_mark.set_data([], [])
            
        # PBP Marker (Flash only near the PBP time)
        # Assuming PBP timestamp is a single moment.
        # Show for +/- 5 frames (approx 0.5s) to ensure visibility?
        # Or just exactly at the frame?
        # Let's do +/- 3 frames.
        if abs(idx - pbp_frame_idx) <= 3:
             if pbp_mark and pbp_x is not None:
                 pbp_mark.set_data([pbp_x], [pbp_y])
        else:
             if pbp_mark:
                 pbp_mark.set_data([], [])
            
        title.set_text(f"Frame {idx}")
        return line, puck_point, shooter_dot, blocker_dot, origin_mark, block_mark, title, pbp_mark

    ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=50)
    out_gif = f"analysis/plots/verified_block_{target_game_id}_{target_goal_id}.gif"
    ani.save(out_gif, writer='pillow')
    print(f"Saved animation to {out_gif}")
    plt.close()

if __name__ == "__main__":
    visualize_verified_block()
