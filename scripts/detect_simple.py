import pandas as pd
import glob
import os

files = glob.glob(r"c:\Users\harri\Desktop\new_puck\data\edge_goals\*positions.csv")
for f in files[:5]:
    df = pd.read_csv(f)
    print(f"File: {os.path.basename(f)}")
    
    # Unit check
    if df['x'].abs().max() > 120:
        df['x'] = (df['x'] - 1200)/12
        df['y'] = -(df['y'] - 510)/12
        
    pucks = df[df['entity_type'] == 'puck']
    goals = pucks[(pucks['x'].abs() > 89) & (pucks['y'].abs() <= 3)]
    
    if not goals.empty:
        gf = goals.iloc[0]['frame_idx']
        last = df['frame_idx'].max()
        print(f"Goal Frame: {gf}, Last: {last}, Extra: {last-gf}")
    else:
        print("No Goal Detected")
