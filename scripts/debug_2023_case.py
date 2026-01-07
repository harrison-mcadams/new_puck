
import sys
import os
import pandas as pd
import numpy as np

# Mock Config
DATA_DIR = r'c:\Users\harri\Desktop\new_puck\data'
ANALYSIS_DIR = r'c:\Users\harri\Desktop\new_puck\analysis'
game_id = '2023020002'
goal_id = '1007'

# Load positions
csv_path = os.path.join(DATA_DIR, 'edge_goals', '20232024', f'game_{game_id}_goal_{goal_id}_positions.csv')
if not os.path.exists(csv_path):
    print(f"File not found: {csv_path}")
    sys.exit(1)

df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} rows.")

puck = df[df['entity_type'] == 'puck']
print(f"Puck rows: {len(puck)}")

# Check speeds
if 'x' in puck.columns and 'timestamp' in puck.columns:
    puck = puck.sort_values('timestamp')
    dt = puck['timestamp'].diff()
    dx = puck['x'].diff()
    dy = puck['y'].diff()
    dist = np.sqrt(dx**2 + dy**2)
    speed = dist / dt # fps? depending on timestamp unit
    
    print(f"Max Speed Raw: {speed.max():.2f}")
    print(f"Timestamps: {puck['timestamp'].head().tolist()}")
