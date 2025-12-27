import pandas as pd
df = pd.read_csv('analysis/league/20252026/5v5/team_summary.csv')
print(df[['team', 'team_goals', 'team_xgs', 'team_xg_per60']].sort_values('team_goals', ascending=False).to_string(index=False))
