import json
import os

def inspect_json():
    path = 'data/edge_goals/20242025/game_2024020202_goal_328_edge.json'
    
    if not os.path.exists(path):
        print("File not found.")
        return
        
    with open(path, 'r') as f:
        data = json.load(f)
        
    if isinstance(data, list):
        print(f"JSON Root is a LIST of length {len(data)}.")
        if len(data) > 0:
            first = data[0]
            print(f"First Item Type: {type(first)}")
            if isinstance(first, dict):
                print(f"First Item Keys: {list(first.keys())}")
                # Check for time/clock keys
                for k in first.keys():
                     if 'time' in k.lower() or 'clock' in k.lower() or 'game' in k.lower():
                        print(f"  > Found Potentital Time Key: {k} = {first[k]}")
            
            # Check Last Item too
            last = data[-1]
            if isinstance(last, dict):
                print("Last Item Time Keys:")
                for k in last.keys():
                     if 'time' in k.lower() or 'clock' in k.lower() or 'game' in k.lower():
                        print(f"  > {k} = {last[k]}")

    elif isinstance(data, dict):
        print(f"JSON Root is a DICT. Keys: {list(data.keys())}")
        # Recurse one level if lists found
        for k, v in data.items():
            if isinstance(v, list) and len(v) > 0:
                print(f"Key '{k}' contains list of length {len(v)}")
                first = v[0]
                if isinstance(first, dict):
                     print(f"  First Item Keys: {list(first.keys())}")

if __name__ == "__main__":
    inspect_json()
