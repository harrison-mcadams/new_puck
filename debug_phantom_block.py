import pandas as pd
import sys
import os
sys.path.append(os.getcwd())
from puck import correction

# Load Data
pbp_path = 'data/20242025/20242025_df.csv'
print(f"Loading {pbp_path}...")
df = pd.read_csv(pbp_path)

# Columns Check
print(f"Columns: {list(df.columns)}")

# Filter Game
game_id = 2024020608
df_game = df[df['game_id'] == game_id].copy()

if df_game.empty:
    print(f"No data for game {game_id}")
    exit()

print(f"Game {game_id} loaded. {len(df_game)} events.")

# Check for synthetic
cols = ['event', 'period', 'period_time', 'total_time_elapsed_s', 'x', 'y']
if 'synthetic' in df.columns: 
    print("Column 'synthetic' found.")
    cols.append('synthetic')
else:
    print("Column 'synthetic' NOT found.")
    
if 'is_synthetic' in df.columns: 
    print("Column 'is_synthetic' found.")
    cols.append('is_synthetic')
else:
    print("Column 'is_synthetic' NOT found.")

# Filter Blocks
blocks = df_game[df_game['event'] == 'blocked-shot'].copy()
print(f"\nBefore Correction (Raw Blocks):")
print(blocks[cols].sort_values('total_time_elapsed_s').to_string())

# Apply Correction
print(f"\nApplying Correction...")
df_game_corr = correction.fix_blocked_shot_attribution(df_game)
blocks_corr = df_game_corr[df_game_corr['event'] == 'blocked-shot']

print(f"\nAfter Correction (Attributed Blocks):")
print(blocks_corr[cols].sort_values('total_time_elapsed_s').to_string())

# Look specifically for the 489.0 phantom
phantom = blocks_corr[blocks_corr['total_time_elapsed_s'] == 489.0]
if not phantom.empty:
    print("\nPHANTOM BLOCK FOUND:")
    print(phantom.T)
else:
    print("\nPhantom block at 489.0 NOT found in corrected data.")
