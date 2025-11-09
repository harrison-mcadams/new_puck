#!/usr/bin/env python3
import numpy as np
import pandas as pd
from analyze import compute_xg_heatmap_from_df
from plot import plot_events

rows = [
    {'game_id':'g1','team_id':101,'home_id':101,'home_abb':'PHI','away_abb':'NYR','x':-30.0,'y':10.0,'xgs':0.2,'event':'shot-on-goal','home_team_defending_side':'right'},
    {'game_id':'g1','team_id':202,'home_id':101,'home_abb':'PHI','away_abb':'NYR','x':20.0,'y':-5.0,'xgs':0.1,'event':'shot-on-goal','home_team_defending_side':'right'},
]

df = pd.DataFrame(rows)
print('DF rows:', len(df))
print(df)

print('\n--- compute_xg_heatmap_from_df team ---')
gx,gy,team_heat,team_xg,team_secs = compute_xg_heatmap_from_df(df, grid_res=5.0, sigma=6.0, normalize_per60=False, selected_team='PHI', selected_role='team')
print('team_heat dtype, shape:', getattr(team_heat,'dtype',None), getattr(team_heat,'shape',None))
print('team_heat nan count:', int(np.isnan(team_heat).sum()), 'sum:', float(np.nansum(team_heat)))
print('team_xg', team_xg)

print('\n--- compute_xg_heatmap_from_df other ---')
gx,gy,other_heat,other_xg,other_secs = compute_xg_heatmap_from_df(df, grid_res=5.0, sigma=6.0, normalize_per60=False, selected_team='PHI', selected_role='other')
print('other_heat dtype, shape:', getattr(other_heat,'dtype',None), getattr(other_heat,'shape',None))
print('other_heat nan count:', int(np.isnan(other_heat).sum()), 'sum:', float(np.nansum(other_heat)))
print('other_xg', other_xg)

print('\n--- plot.plot_events heatmaps ---')
fig, ax, heatmaps = plot_events(df, events_to_plot=['xgs'], return_heatmaps=True, heatmap_split_mode='team_not_team', team_for_heatmap='PHI')
print('heatmaps keys:', list(heatmaps.keys()))
for k,v in heatmaps.items():
    print(k, type(v), getattr(v,'dtype',None), getattr(v,'shape',None), 'nan count', int(np.isnan(v).sum()), 'sum', float(np.nansum(v)))

print('\nDone')

