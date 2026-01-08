
import json

def get_stats(data, season, team):
    team_data = data.get(season, {}).get(team, {})
    if not team_data:
        return None
    x_adj = team_data.get('x', {})
    # Return a few key points
    return {
        '20': x_adj.get('20'),
        '50': x_adj.get('50'),
        '85': x_adj.get('85')
    }

path = "data/arena_adjustments.json"
with open(path, 'r') as f:
    data = json.load(f)

seasons = sorted(data.keys())
teams = ['Lightning', 'Rangers']

print(f"{'Season':<10} | {'Team':<10} | {'X=20':<8} | {'X=50':<8} | {'X=85':<8}")
print("-" * 55)

for s in seasons:
    for t in teams:
        stats = get_stats(data, s, t)
        if stats:
            print(f"{s:<10} | {t:<10} | {str(stats['20']):<8} | {str(stats['50']):<8} | {str(stats['85']):<8}")
