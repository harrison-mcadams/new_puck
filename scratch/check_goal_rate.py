
import json
with open('analysis/mixed_effects_heatmaps_20252026/team_stats_summary.json', 'r') as f:
    data = json.load(f)

total_seconds = 0
total_goals = 0
total_games = 0
for team, stats in data.items():
    if '5v5' in stats:
        total_seconds += stats['5v5']['seconds']
        total_goals += stats['5v5']['goals_for']
        total_games += stats['5v5']['games_played']

print(f"Total Seconds: {total_seconds}")
print(f"Total Goals: {total_goals}")
print(f"Total Games (Team-Games): {total_games}")
print(f"Total Games (Unique Estimate): {total_games / 2}")
print(f"Combined G/60: {(total_goals / (total_seconds / 2)) * 3600}")
print(f"Per-Team G/60: {(total_goals / total_seconds) * 3600}")
