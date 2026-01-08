
import json
import os

path = "data/arena_adjustments.json"
if not os.path.exists(path):
    print("File not found")
    exit()

with open(path, 'r') as f:
    data = json.load(f)

biases = []
for season, arenas in data.items():
    for arena, ax_data in arenas.items():
        for axis in ['x', 'y']:
            for bucket, val in ax_data.get(axis, {}).items():
                biases.append({
                    'season': season,
                    'arena': arena,
                    'axis': axis,
                    'bucket': bucket,
                    'val': val
                })

# Sort by absolute value
biases.sort(key=lambda x: abs(x['val']), reverse=True)

print("Top 10 Largest Biases (Absolute Value):")
for i, b in enumerate(biases[:10]):
    print(f"{i+1}. {b['arena']} ({b['season']}) | {b['axis']} @ {b['bucket']}ft: {b['val']:.2f} ft")
