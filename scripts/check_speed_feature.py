
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Check latest season
path = Path('data/20252026/20252026_df.csv')
if not path.exists():
    print("2025 data not found")
    sys.exit(0)

print(f"Loading {path}...")
df = pd.read_csv(path)

if 'speed_from_last_event' not in df.columns:
    print("Feature speed_from_last_event NOT FOUND")
    sys.exit(1)

stats = df['speed_from_last_event'].describe()
print("\nFeature Stats:")
print(stats)

# Check distribution for shots vs other events
shots = df[df['event'].isin(['shot-on-goal', 'goal'])]
print("\nShot Speed Stats:")
print(shots['speed_from_last_event'].describe())

# Save histogram
plt.figure()
shots['speed_from_last_event'].hist(bins=50, range=(0, 100))
plt.title("Speed From Last Event (Shots)")
plt.savefig('analysis/speed_feature_dist.png')
print("Saved histogram to analysis/speed_feature_dist.png")
