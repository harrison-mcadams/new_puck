
import sys
import os
import pandas as pd
import numpy as np

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from puck import timing

def inspect_event():
    game_id = 2025020021
    target_time = 3445.0 # 02:35 P3 (approx)
    
    print(f"Inspecting Game {game_id} at Time {target_time}s")
    
    # 1. Load Shifts
    df_shifts = timing._get_shifts_df(game_id, season='20252026')
    if df_shifts.empty:
        print("No shifts found!")
        return

    # 2. Get Shifts Active at this time
    # Filter: start <= target <= end
    mask_active = (df_shifts['start_total_seconds'] <= target_time) & (df_shifts['end_total_seconds'] >= target_time)
    active_shifts = df_shifts[mask_active].copy()
    
    # 3. Classify Roles (G vs S)
    classification = timing._classify_player_roles(df_shifts)
    roles = classification['roles']
    
    # 4. Count Skaters per Team
    teams = active_shifts['team_id'].unique()
    
    print("\n--- Active Players on Ice ---")
    for tid in teams:
        team_shifts = active_shifts[active_shifts['team_id'] == tid]
        skaters = []
        goalies = []
        
        for _, row in team_shifts.iterrows():
            pid = str(row['player_id'])
            # name = row['raw']['player']['fullName'] if 'player' in row['raw'] else pid
            # Simpler name extraction if possible, else use ID
            name = pid
            try:
                if 'firstName' in row['raw'] and 'lastName' in row['raw']:
                     name = f"{row['raw']['firstName']} {row['raw']['lastName']}"
            except: pass
            
            role = roles.get(pid, 'S')
            if role == 'G':
                goalies.append(name)
            else:
                skaters.append(name)
        
        print(f"\nTeam {tid}:")
        print(f"  Goalies ({len(goalies)}): {goalies}")
        print(f"  Skaters ({len(skaters)}): {skaters}")

    # 5. Check API Event Context
    print("\n--- API Event Context ---")
    csv_path = 'data/20252026/20252026.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, low_memory=False)
        df_g = df[df['game_id'] == game_id]
        
        # Find event near target time
        # tolerance 
        mask_event = (df_g['total_time_elapsed_seconds'] >= target_time - 2) & (df_g['total_time_elapsed_seconds'] <= target_time + 2)
        events = df_g[mask_event]
        
        print(events[['period_time', 'period', 'event', 'game_state', 'is_net_empty', 'total_time_elapsed_seconds']])
        
    else:
        print("CSV not found.")

if __name__ == '__main__':
    inspect_event()
