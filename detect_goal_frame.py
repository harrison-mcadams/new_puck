import pandas as pd
import glob
import os

# Game 2024020608, Goal 307
season = '20242025'
game_id = 2024020608
goal_id = 307

edge_dir = f'data/edge_goals/{season}'
pattern_csv = f"{edge_dir}/game_{game_id}_goal_{goal_id}_positions.csv"
csv_files = glob.glob(pattern_csv)

if not csv_files:
    print("File not found.")
    exit()

df = pd.read_csv(csv_files[0])
puck = df[df['entity_type'] == 'puck'].copy()

# Normalize X for checking
# Goal lines are at +/- 89
# Check for crossing 89
puck['abs_x'] = puck['x'].abs()

# Find frames where puck is IN THE NET (x > 89, y between -3 and 3)
in_net = puck[
    (puck['abs_x'] > 89.0) & 
    (puck['y'] > -3.0) & 
    (puck['y'] < 3.0)
]

print(f"Total Frames: {len(puck)}")

if not in_net.empty:
    first_net_frame = in_net['frame_idx'].min()
    print(f"Goal Logic Triggered at Frame: {first_net_frame}")
    print(f"Time Offset (assuming 0.1s/frame): {first_net_frame * 0.1:.2f}s")
    
    # Check what % of clip is pre-goal
    pre_goal_pct = first_net_frame / len(puck)
    print(f"Pre-Goal portion: {pre_goal_pct*100:.1f}%")
else:
    print("Puck never detected inside net coordinates.")
    # Maybe check for proximity if it hits post or something
    closest = puck.loc[puck['abs_x'].idxmax()]
    print(f"Closest X: {closest['x']} at Frame {closest['frame_idx']}")
