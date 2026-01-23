import json
import numpy as np

s = json.load(open('analysis/mixed_effects_heatmaps_20252026/team_stats_summary.json'))

print("=== DEFENSIVE PROFILES (5v5) ===")
print(f"{'Team':<5} {'CA/60':<8} {'xGA/60':<8} {'xG/Shot':<8} {'GA/60':<8}")

teams = sorted(s.keys())
xg_per_shot_list = []

for t in teams:
    d = s[t]['5v5']
    toi = d['seconds']/3600
    if toi < 10: continue
    
    ca = d['attempts_against']
    xga = d['xg_against']
    ga = d['goals_against']
    
    ca60 = ca / toi
    xga60 = xga / toi
    ga60 = ga / toi
    xg_per_shot = xga / ca if ca > 0 else 0
    
    xg_per_shot_list.append(xg_per_shot)
    
    if t in ['NYR', 'EDM', 'TOR', 'COL', 'FLA', 'SJS']:
        print(f"{t:<5} {ca60:<8.1f} {xga60:<8.2f} {xg_per_shot:<8.4f} {ga60:<8.2f}")

avg_xgs = np.mean(xg_per_shot_list)
print(f"\nLeague Avg xG/Shot Against: {avg_xgs:.4f}")

edm_xgs = s['EDM']['5v5']['xg_against'] / s['EDM']['5v5']['attempts_against']
print(f"EDM vs League: {edm_xgs / avg_xgs:.2f}x")

if edm_xgs < avg_xgs:
    print("=> EDM suppresses shot quality (lower xG/shot than avg)")
else:
    print("=> EDM allows high quality shots (higher xG/shot than avg)")
