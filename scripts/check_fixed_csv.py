
import pandas as pd
import sys

csv_path = 'data/20252026/20252026_df.csv'
print(f"Loading {csv_path}...")
try:
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows.")
except Exception as e:
    print(f"Failed: {e}")
    sys.exit(1)

# Check game 2025020460
target_game_id = 2025020460
print(f"Checking game {target_game_id}...")
g = df[df['game_id'] == target_game_id]
print(f"Found {len(g)} rows for game {target_game_id}.")

# Check for specific shot x=73, y=9 (approx)
matches = g[ (g['x'] - 73.0).abs() < 1.0 ]
matches = matches[ (matches['y'] - 9.0).abs() < 1.0 ]
matches = matches[ matches['period'] == 1 ]
print(f"Found {len(matches)} specific matching events.")

if not matches.empty:
    print(matches[['event', 'period', 'period_time', 'time_elapsed_in_period_s', 'total_time_elapsed_s']].to_string())
else:
    print("No matches found.")

# Check overall stats for this game
print("\nStats for game 2025020181:")
print(g[['period_time', 'time_elapsed_in_period_s']].isna().sum())
