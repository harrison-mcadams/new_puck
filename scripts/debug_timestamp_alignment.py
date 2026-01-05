import pandas as pd
import numpy as np
import os
import sys

def debug_alignment():
    # Target: Game 2024020202, Goal 328
    csv_path = 'data/edge_goals/20242025/game_2024020202_goal_328_positions.csv'
    
    if not os.path.exists(csv_path):
        print("CSV not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    # 1. Goal Frame Detection (My current logic)
    df_puck = df[df['entity_type'] == 'puck'].sort_values('frame_idx')
    
    # Check coords
    if df_puck['x'].abs().max() > 200:
        df_puck['x'] = (df_puck['x'] - 1200.0) / 12.0
        df_puck['y'] = -(df_puck['y'] - 510.0) / 12.0
        
    in_net = df_puck[(df_puck['x'].abs() > 89.0) & (df_puck['y'].abs() < 6.0)]
    
    detected_goal_frame = in_net['frame_idx'].min() if not in_net.empty else df_puck['frame_idx'].max()
    print(f"Detected Goal Frame: {detected_goal_frame}")
    
    # Get Timestamp of Goal Frame
    goal_row = df_puck[df_puck['frame_idx'] == detected_goal_frame]
    if goal_row.empty:
        print("Goal frame not found in puck data.")
        return
        
    goal_ts = goal_row['timestamp'].iloc[0]
    print(f"Goal Frame Timestamp: {goal_ts}")
    
    # 2. Frame Interval Check
    timestamps = sorted(df_puck['timestamp'].unique())
    diffs = np.diff(timestamps)
    mean_diff = np.mean(diffs)
    print(f"Mean Frame Interval (Timestamp Diff): {mean_diff:.2f}")
    
    # 3. PBP Offset Calculation
    # PBP Goal Time: 14:12 (852s elapsed)
    # PBP Block Time: 14:04 (844s elapsed)
    # Diff: 8.0 seconds
    
    pbp_diff_s = 8.0
    
    # Calculate Target Timestamp for Block (Backwards from Goal)
    # If timestamps are in milliseconds or nanoseconds?
    # Previous check showed '17308603928'. 
    # Let's assume the diff '1' meant something else? Or was it 150ms?
    # Wait, previous diff was '1'. That is extremely small if standard unix.
    # But user saw 30 frame offset (~1s at 30fps).
    
    # Let's re-verify the UNITS of this diff.
    # If diff is ~1500000 (nanos) or 150 (ms)?
    
    # Let's project BACKWARDS 8 seconds from Goal Timestamp
    # Problem: Converting "8 seconds" to "Timestamp Units" without knowing the unit.
    # Solution: Use the Frame Interval we just calculated.
    
    # Frames to go back = 8.0s / (Interval_in_Seconds)
    # But we don't know Interval in Seconds definitively yet (no clock).
    # BUT, we can convert "8 Seconds" into "Frames" assuming standard Edge FPS?
    # User said "About frame 15" vs "Frame 45".
    # 30 frame diff.
    # If 30 frames = 3.6s mismatch -> 1 frame = 0.12s (approx 8.35 Hz)
    # If 30 frames = 3.6s -> 120ms/frame.
    
    # Let's find the frame corresponding to "Goal Timestamp - (8.0 * (Timestamp_Units_Per_Second))"
    # We estimate Timestamp_Units_Per_Second from the data if possible?
    # No, we can't without clock.
    
    # ALTERNATIVE:
    # Use the PBP "TimeInPeriod" vs "Timestamp"? No linkage.
    
    # Let's look at the Block Frame I FOUND (The Min Dist one).
    # See what its timestamp is.
    
    # Block Logic (from visualize scipt)
    blocker_id = 8476473 # Kulikov
    col_id = 'player_id' if 'player_id' in df.columns else 'entity_id'
    
    # Force numeric
    if col_id in df.columns:
         df[col_id] = pd.to_numeric(df[col_id], errors='coerce')
    
    # Search for Blocker
    if float(blocker_id) in df[col_id].values:
        print(f"Blocker {blocker_id} FOUND in data.")
        df_blocker = df[df[col_id] == float(blocker_id)]
    elif str(blocker_id) in df[col_id].astype(str).values:
         print(f"Blocker {blocker_id} FOUND (as string).")
         df_blocker = df[df[col_id].astype(str) == str(blocker_id)]
    else:
         print(f"Blocker {blocker_id} NOT FOUND.")
         # print(f"IDs in file: {df[col_id].unique()}") # omit huge list
         return

    merged = pd.merge(df_puck, df_blocker[['frame_idx', 'x', 'y']], on='frame_idx', suffixes=('_puck', '_blocker'))
    
    if merged.empty:
        print("No intersecting frames.")
        return

    merged['dist'] = np.sqrt((merged['x_puck'] - merged['x_blocker'])**2 + (merged['y_puck'] - merged['y_blocker'])**2)
    
    min_row = merged.loc[merged['dist'].idxmin()]
    found_block_frame = min_row['frame_idx']
    found_block_ts = min_row['timestamp_puck'] if 'timestamp_puck' in min_row else df_puck[df_puck['frame_idx']==found_block_frame]['timestamp'].iloc[0]
    
    print(f"My Logic 'Found' Block Frame: {found_block_frame} (Dist {min_row['dist']:.2f})")
    print(f"Found Block Timestamp: {found_block_ts}")
    
    ts_diff = goal_ts - found_block_ts
    print(f"Timestamp Delta (Goal - Block): {ts_diff}")
    
    # Ratio:
    # We expect this delta to represent 8.0 seconds.
    # So 1 Second = ts_diff / 8.0 (?)
    # If my block IS correct.
    # But User says my block is WRONG (it is "Wrong Conclusion").
    # User implies PBP is correct. 
    # So Real Block Frame SHOULD be 8.0s before Goal.
    
    # Let's look at the "Frame 15" the user mentioned.
    # If Block is at frame 15 (hypothetically).
    # Goal is at 89?
    # 89 - 15 = 74 frames.
    # 74 frames = 8 seconds?
    # -> 1 frame = 0.108s (~9.2 Hz).
    
    # Let's check Frame 45 (My block?).
    # 89 - 45 = 44 frames.
    
    # Check what is happening at Frame 15 (or thereabouts).
    # Is the puck near Kulikov?
    
    if 15 in merged['frame_idx'].values:
        row_15 = merged[merged['frame_idx'] == 15].iloc[0]
        print(f"At Frame 15: Dist to Kulikov = {row_15['dist']:.2f}")
    
    # Check other local minima?
    # Maybe there is a secondary close encounter?
    
    # Also Check: Is '89' really the goal?
    # Or is Goal '125' like I hypothesized?
    # User said "actual block occurs around frame 45".
    # Wait, User said: "it [PBP Marker?] appears on about frame 15 when the actual block occurs around frame 45."
    # User says ACTUAL block is frame 45.
    # MY SCRIPT found block at frame 45 (approx).
    # My "Lag" calc said PBP was 3.6s early.
    # User says "PBP is correct".
    # This implies my TIME ALIGNMENT is wrong.
    # If Block is 45, and PBP says Block is 8s before goal.
    # Then Goal MUST be at 45 + (8s in frames).
    # If 10Hz, Goal = 45 + 80 = 125.
    
    # Does Goal happen at 125?
    # I detected Goal at 89.
    # Let's check Frame 125 coordinates.
    
    frame_125_row = df_puck[df_puck['frame_idx'] == 125]
    if not frame_125_row.empty:
        print(f"Frame 125 Coords: ({frame_125_row['x'].iloc[0]:.2f}, {frame_125_row['y'].iloc[0]:.2f})")
    else:
        print("Frame 125 not found.")

if __name__ == "__main__":
    debug_alignment()
