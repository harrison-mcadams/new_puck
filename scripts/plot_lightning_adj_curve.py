
import json
import matplotlib.pyplot as plt
import numpy as np

path = "data/arena_adjustments.json"
with open(path, 'r') as f:
    data = json.load(f)

# Tampa 2018-2019
x_adj = data.get('20182019', {}).get('Lightning', {}).get('x', {})
sorted_x = sorted([int(k) for k in x_adj.keys()])
deltas = [x_adj[str(x)] for x in sorted_x]

plt.figure(figsize=(10, 6))
plt.plot(sorted_x, deltas, marker='o', markersize=2, linestyle='-', label='Raw Adjustment')
plt.axhline(0, color='red', linestyle='--')
plt.title("Tampa Bay Lightning (2018-2019) X-Adjustment Curve")
plt.xlabel("Recorded X Coordinate (Absolute)")
plt.ylabel("Adjustment (Feet)")
plt.grid(True, alpha=0.3)

# Highlight the spike
plt.annotate('The 14ft Spike', xy=(85, 14), xytext=(60, 16),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.savefig("lightning_adj_curve.png")
print("Saved lightning_adj_curve.png")
