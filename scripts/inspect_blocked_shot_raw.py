import pandas as pd
import os
import numpy as np

def inspect_season(season, label):
    path = f"data/{season}/{season}_df.csv"
    # Fallback to specific file if generic not found (e.g. _df might differ)
    if not os.path.exists(path):
        # Try finding any csv
        if os.path.exists(f"data/{season}"):
            files = [f for f in os.listdir(f"data/{season}") if f.endswith('.csv')]
            if files:
                path = f"data/{season}/{files[0]}"
    
    if not os.path.exists(path):
        print(f"Skipping {label} ({season}): File not found at {path}")
        return

    print(f"\n--- Inspecting {label} ({season}) ---")
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return
    
    # Filter Blocked Shots
    mask = df['event'].astype(str).str.strip().str.lower() == 'blocked-shot'
    df_blocked = df[mask]
    
    if df_blocked.empty:
        print("No blocked shots found.")
        return

    # Sample randomly
    if len(df_blocked) > 5:
        sample = df_blocked.sample(5, random_state=42)
    else:
        sample = df_blocked
    
    # Columns of interest - standardized
    cols_priority = ['game_id', 'period', 'period_time', 'home_abb', 'away_abb', 'home_team_defending_side', 
                     'team_id', 'home_id', 'away_id', 'x', 'y', 'description']
    
    # Filter for existing columns
    use_cols = [c for c in cols_priority if c in df.columns]
    
    print(sample[use_cols].to_string())
    
    # Detailed Analysis of First Sample Row
    print("\n[Deep Dive Analysis of First Sample Row]")
    row = sample.iloc[0]
    
    tid = row.get('team_id')
    hid = row.get('home_id')
    aid = row.get('away_id')
    
    # Determine Role (Home/Away)
    role = "UNKNOWN"
    try:
        if str(int(float(str(tid)))) == str(int(float(str(hid)))): role = "HOME"
        elif str(int(float(str(tid)))) == str(int(float(str(aid)))): role = "AWAY"
    except:
        pass
    
    print(f"  Event Team ID: {tid} is {role}")
    print(f"  Coordinates: ({row.get('x')}, {row.get('y')})")
    
    def_side = row.get('home_team_defending_side', 'Unknown')
    print(f"  Home Defending Side: {def_side}")
    
    # Logical Deduction
    # Zone Definitions (Standard NHL Rink: X from -100 to 100)
    # Defending Zone is typically where the goalie is. 
    # If Home Defends LEFT (-X), then -X is Home's Defensive Zone.
    
    conclusion = "INCONCLUSIVE"
    
    if isinstance(def_side, str) and def_side.lower() in ['left', 'right']:
        side_norm = def_side.lower()
        x_val = row.get('x')
        
        if pd.notna(x_val):
            # 1. Identify Zone of Event
            zone = "NEUTRAL"
            if x_val < -25: zone = "LEFT_ZONE"
            elif x_val > 25: zone = "RIGHT_ZONE"
            
            # 2. Identify Owner's Defensive Zone
            home_def_zone = "LEFT_ZONE" if side_norm == 'left' else "RIGHT_ZONE"
            away_def_zone = "RIGHT_ZONE" if side_norm == 'left' else "LEFT_ZONE" # Opposing
            
            print(f"  Event Zone: {zone}")
            print(f"  Home Def Zone: {home_def_zone}")
            print(f"  Away Def Zone: {away_def_zone}")
            
            if role == "HOME":
                if zone == home_def_zone:
                    conclusion = "BLOCKER (Defending Own Zone)"
                elif zone == away_def_zone:
                    conclusion = "SHOOTER (Attacking Opponent Zone)"
                else:
                    conclusion = "NEUTRAL ZONE BLOCK (Likely Blocker)"
            elif role == "AWAY":
                if zone == away_def_zone:
                    conclusion = "BLOCKER (Defending Own Zone)"
                elif zone == home_def_zone:
                    conclusion = "SHOOTER (Attacking Opponent Zone)"
                else:
                    conclusion = "NEUTRAL ZONE BLOCK (Likely Blocker)"
                    
    print(f"  => CONCLUSION: Team {tid} appears to be the: **{conclusion}**")

if __name__ == "__main__":
    inspect_season('20252026', 'Current Season')
    inspect_season('20192020', 'Past Season (2019-2020)')
