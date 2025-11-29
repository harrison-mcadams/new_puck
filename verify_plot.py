
import pandas as pd
import numpy as np
import analyze
import os
import matplotlib.pyplot as plt

# Mock DataFrame
data = {
    'game_id': [2025020001, 2025020001, 2025020002, 2025020002],
    'home_abb': ['ANA', 'ANA', 'BOS', 'BOS'],
    'away_abb': ['BOS', 'BOS', 'ANA', 'ANA'],
    'home_id': [1, 1, 2, 2],
    'away_id': [2, 2, 1, 1],
    'team_id': [1, 2, 2, 1], # ANA, BOS, BOS, ANA
    'x': [50, -50, 60, -60],
    'y': [0, 0, 10, -10],
    'xgs': [0.5, 0.2, 0.6, 0.3],
    'event': ['Shot', 'Shot', 'Shot', 'Shot'],
    'home_team_defending_side': ['left', 'left', 'right', 'right'],
    'total_time_elapsed_seconds': [100, 200, 300, 3600] # Simulate game time
}
df = pd.DataFrame(data)

# Create output directory
out_dir = 'static/test_plot'
os.makedirs(out_dir, exist_ok=True)

# Set season attribute on df
df.season = '20252026_test'

# Run xg_maps_for_season
print("Running xg_maps_for_season with mock data...")
try:
    analyze.xg_maps_for_season(
        season_or_df=df,
        out_dir=out_dir,
        min_events=1, # Lower min_events to ensure our small data is processed
        grid_res=5.0 # Coarse grid for speed
    )
    print("Function completed successfully.")
except Exception as e:
    print(f"Function failed: {e}")
    import traceback
    traceback.print_exc()

# Check if plot exists
plot_path = os.path.join(out_dir, '20252026_test', '20252026_test_season_comparison.png')
if os.path.exists(plot_path):
    print(f"Plot created at {plot_path}")
else:
    print(f"Plot NOT found at {plot_path}")
