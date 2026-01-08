
import json

def report_arena(data, season, arena_key):
    arena_data = data.get(season, {}).get(arena_key, {})
    if not arena_data:
        print(f"No data for {arena_key} in {season}")
        return
    
    x_adj = arena_data.get('x', {})
    y_adj = arena_data.get('y', {})
    
    # Check X at key distances: 20ft from net (X=69), 50ft from net (X=39), and near net (X=85)
    # Net is at 89. abs(x) is distance from center.
    # 0 = center, 89 = net.
    
    points = ["20", "40", "60", "70", "80", "85", "89"]
    print(f"\n--- {arena_key} ({season}) X-Adjustment ---")
    print("X Coord | Adjustment (ft)")
    print("--------|----------------")
    for p in points:
        val = x_adj.get(p, "N/A")
        print(f"{p.ljust(7)} | {val}")

path = "data/arena_adjustments.json"
with open(path, 'r') as f:
    data = json.load(f)

for season in ["20182019", "20232024"]:
    print(f"\n{'='*10} SEASON: {season} {'='*10}")
    report_arena(data, season, "Lightning")
    report_arena(data, season, "Rangers")
