
import pandas as pd
import numpy as np

df = pd.read_csv('data/20252026.csv')
shot_events = ['shot-on-goal', 'goal', 'missed-shot', 'blocked-shot']
df_shots = df[df['event'].isin(shot_events)].copy()

# Print top 10 xG shots
print("--- Top 10 xG Shots ---")
cols = ['event', 'x', 'y', 'distance', 'angle_deg', 'xgs', 'home_team_defending_side']
# Filter for existing columns
cols = [c for c in cols if c in df_shots.columns]
print(df_shots.sort_values('xgs', ascending=False)[cols].head(10))

# Print stats
print("\n--- Correlation x vs xgs ---")
if 'x' in df_shots.columns and 'xgs' in df_shots.columns:
    print(df_shots[['x', 'xgs']].corr())

print("\n--- x Ranges ---")
if 'x' in df_shots.columns:
    print(df_shots['x'].describe())

print("\n--- Event xG averages ---")
if 'xgs' in df_shots.columns:
    print(df_shots.groupby('event')['xgs'].mean())
