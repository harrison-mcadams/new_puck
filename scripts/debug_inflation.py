
import json
import pandas as pd
import os

path = 'analysis/league/20252026/5v5/20252026_team_summary.json'
try:
    with open(path, 'r') as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    print("Columns:", df.columns.tolist())
    
    # Calculate Rates
    # Seconds -> Minutes
    df['min'] = df['team_seconds'] / 60.0
    
    # Rates per 60
    df['calc_xGF60'] = (df['team_xgs'] / df['min']) * 60
    df['calc_GF60'] = (df['team_goals'] / df['min']) * 60
    
    # Display
    # Use available columns
    cols = ['team', 'team_seconds', 'team_xgs', 'team_goals', 'calc_xGF60', 'calc_GF60']
    cols = [c for c in cols if c in df.columns] + ['min']
    print(df[cols].round(2).to_markdown(index=False))
    
    print("\nLeague Totals:")
    total_xg = df['team_xgs'].sum()
    total_goals = df['team_goals'].sum()
    print(f"Total xG: {total_xg:.2f}")
    print(f"Total Goals: {total_goals:.2f}")
    print(f"Ratio (xG/Goals): {total_xg/total_goals:.3f}")

except Exception as e:
    print(e)
