
import json
import os

ADJUSTMENTS_FILE = os.path.join("data", "arena_adjustments.json")

def inspect_keys():
    if not os.path.exists(ADJUSTMENTS_FILE):
        print(f"File not found: {ADJUSTMENTS_FILE}")
        return

    try:
        with open(ADJUSTMENTS_FILE, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    print(f"Seasons found: {list(data.keys())}")
    
    # Check latest season
    latest_season = max(data.keys())
    print(f"\nChecking Season: {latest_season}")
    
    arenas = sorted(list(data[latest_season].keys()))
    print(f"Arenas ({len(arenas)}):")
    for a in arenas:
        print(f"  - '{a}'")
        
    # Sample logic test
    sample_arena = arenas[0]
    print(f"\nSample Data for '{sample_arena}':")
    print(f"  X keys: {list(data[latest_season][sample_arena].get('x', {}).keys())[:5]}...")
    print(f"  Y keys: {list(data[latest_season][sample_arena].get('y', {}).keys())[:5]}...")

if __name__ == "__main__":
    inspect_keys()
