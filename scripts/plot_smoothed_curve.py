
import json
import matplotlib.pyplot as plt
import numpy as np

# Load the NEW smoothed data
path = "data/arena_adjustments.json"
with open(path, 'r') as f:
    data = json.load(f)

x_adj_smoothed = data.get('20182019', {}).get('Lightning', {}).get('x', {})
sorted_x = sorted([int(k) for k in x_adj_smoothed.keys()])
deltas_smoothed = [x_adj_smoothed[str(x)] for x in sorted_x]

# Approximate the RAW data based on my previous extraction (the 14ft spike was at 85)
# I don't have the full raw JSON anymore, so I'll create a mockup of the 'jagged' nature for the visual comparison
# or I can just plot the smoothed one and describe the change. 
# Better: I'll just plot the smoothed curve and point out the peak is now ~10.9.

plt.figure(figsize=(10, 6))
plt.plot(sorted_x, deltas_smoothed, marker='o', markersize=2, linestyle='-', color='blue', label='Smoothed Adjustment (Window=7)')
plt.axhline(0, color='red', linestyle='--')

plt.title("Tampa Bay Lightning (2018-2019) X-Adjustment Curve (SMOOTHED)")
plt.xlabel("Recorded X Coordinate (Absolute)")
plt.ylabel("Adjustment (Feet)")
plt.grid(True, alpha=0.3)

# Annotate the new peak
plt.annotate('Smoothed Peak (~10.9ft)', xy=(85, 10.86), xytext=(60, 13),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.legend()
plt.savefig("lightning_adj_curve_smoothed.png")
print("Saved lightning_adj_curve_smoothed.png")
