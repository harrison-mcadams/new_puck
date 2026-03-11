import json
import pandas as pd

with open(r'C:\Users\harri\Desktop\new_puck\analysis\mixed_effects_heatmaps_20252026\team_stats_summary.json', 'r') as f:
    stats = json.load(f)
    
teams = ['PHI', 'COL']
states = ['5v5', '5v4', '4v5']

data = []
for team in teams:
    for state in states:
        team_stats = stats.get(team, {}).get(state, {})
        seconds = team_stats.get('seconds', 0)
        
        if seconds > 0:
            xg_for = team_stats.get('xg_for', 0)
            goals_for = team_stats.get('goals_for', 0)
            
            p60 = 3600.0 / seconds
            xg_60 = xg_for * p60
            goals_60 = goals_for * p60
            
            data.append({
                'Team': team,
                'State': state,
                'xtG/60': round(xg_60, 2),
                'Actual GF/60': round(goals_60, 2),
                'Diff': round(goals_60 - xg_60, 2)
            })

df = pd.DataFrame(data)
print(df.to_markdown(index=False))
