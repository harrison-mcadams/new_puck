
import pandas as pd
df = pd.read_csv('analysis/league/20252026/5v5/20252026_team_summary.csv')
total_goals = df['team_goals'].sum()
total_seconds = df['team_seconds'].sum()
total_games = df['n_games'].sum()
print(f"Total 5v5 Goals: {total_goals}")
print(f"Total 5v5 Seconds (Team-Seconds): {total_seconds}")
print(f"Total Team-Games: {total_games}")
print(f"Total Unique Games (Estimate): {total_games / 2}")
print(f"Total 5v5 Time (Hours): {total_seconds / 2 / 3600}")
print(f"Avg 5v5 Time per Game (Mins): {(total_seconds / 2 / (total_games / 2)) / 60}")
print(f"Empiric 5v5 Goal Rate (Combined G/60): {(total_goals / (total_seconds / 2)) * 3600}")
print(f"Empiric 5v5 Goal Rate (Per-Team G/60): {(total_goals / total_seconds) * 3600}")
