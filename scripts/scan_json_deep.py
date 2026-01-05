import json
import os
from collections import Counter

def deep_scan_json():
    path = 'data/edge_goals/20242025/game_2024020202_goal_328_edge.json'
    
    if not os.path.exists(path):
        print("File not found.")
        return
        
    print(f"Scanning {path}...")
    with open(path, 'r') as f:
        data = json.load(f)
        
    all_keys = Counter()
    time_values = []
    
    def recurse(obj, depth=0):
        if isinstance(obj, dict):
            for k, v in obj.items():
                all_keys[k] += 1
                # Check value for game clock format (e.g. "14:02")
                if isinstance(v, str) and ':' in v and len(v) < 6:
                     # heuristic for "MM:SS"
                     if v[0].isdigit() and v[-1].isdigit():
                         time_values.append((k, v))
                
                recurse(v, depth+1)
        elif isinstance(obj, list):
            for item in obj:
                recurse(item, depth+1)

    recurse(data)
    
    print("\n--- All Unique Keys Found ---")
    for k, count in all_keys.most_common():
        print(f"{k}: {count}")
        
    print("\n--- Potential Game Clock Values ---")
    # Show first 10 unique time values
    unique_times = sorted(list(set(time_values)))
    for k, v in unique_times[:20]:
        print(f"Key: {k} | Value: {v}")

    if not unique_times:
        print("No 'MM:SS' strings found.")

if __name__ == "__main__":
    deep_scan_json()
