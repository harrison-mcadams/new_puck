
import json
import pandas as pd

def league_stats():
    with open('analysis/mixed_effects_heatmaps_20252026/team_stats_summary.json', 'r') as f:
        data = json.load(f)
    
    totals = {}
    for team, stats in data.items():
        for state, metrics in stats.items():
            if state not in totals:
                totals[state] = {'seconds': 0, 'goals': 0, 'xg': 0, 'xg_mixed': 0}
            
            totals[state]['seconds'] += metrics.get('seconds', 0)
            totals[state]['goals'] += metrics.get('goals_for', 0)
            totals[state]['xg'] += metrics.get('xg_for', 0)
            totals[state]['xg_mixed'] += metrics.get('xg_mixed_for', metrics.get('xg_for', 0)) # Fallback
            
    print(f"{'State':<10} | {'G/60':<10} | {'xG/60 (Base)':<12} | {'xG/60 (Mixed)':<12}")
    print("-" * 55)
    for state in ['5v5', '5v4', '4v5']:
        s = totals[state]
        if s['seconds'] == 0: continue
        
        g60 = (s['goals'] / s['seconds']) * 3600
        xg60 = (s['xg'] / s['seconds']) * 3600
        xgm60 = (s['xg_mixed'] / s['seconds']) * 3600
        print(f"{state:<10} | {g60:<10.2f} | {xg60:<12.2f} | {xgm60:<12.2f}")

if __name__ == "__main__":
    league_stats()
