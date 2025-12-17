
import sys
import os
import pandas as pd
import numpy as np

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import timing
# from puck.analyze import _apply_intervals # Removed unused import

def calc_player_goals(season='20252026', player_id=8484762, player_name="Beckett Sennecke"):
    print(f"--- Calculating 5v5 Goals for {player_name} ({player_id}) ---")
    
    # 1. Load Data
    print("Loading Season Data...")
    df_data = timing.load_season_df(season)
    
    # 2. Find Games
    # We find games where this player has at least one event OR shift.
    # Shifts are more reliable for "on roster".
    # But filtering DF by player_id is faster first pass.
    player_events = df_data[df_data['player_id'] == player_id]
    game_ids = sorted(player_events['game_id'].unique())
    
    print(f"Found {len(game_ids)} games with events for player.")
    
    total_gf = 0
    total_ga = 0
    collected_goals = []
    
    for gid in game_ids:
        # Get Game DF
        df_game = df_data[df_data['game_id'] == gid].copy()
        
        # Determine Player's Team in this game
        # Usually from events
        p_events = df_game[df_game['player_id'] == player_id]
        if p_events.empty:
            # Try to get team from shifts if events empty (unlikely if he played)
            # Or just skip
            pass

        # Try to infer team_id from shifts if p_events is empty
        # But usually p_events has at least one item if he played.
        # If not, get team from shifts
        if p_events.empty:
            # Fallback
            df_shifts_raw = timing._get_shifts_df(int(gid), season=season)
            p_s = df_shifts_raw[df_shifts_raw['player_id'] == player_id]
            if not p_s.empty:
                team_id = p_s.iloc[0]['team_id']
            else:
                continue
        else:
             team_id = p_events.iloc[0]['team_id']

        # Get Player Shifts (Time Ranges)
        df_shifts = timing._get_shifts_df(int(gid), season=season)
        if df_shifts.empty:
            continue
            
        p_shifts = df_shifts[df_shifts['player_id'] == int(player_id)]
        # List of (start, end)
        shift_intervals = list(zip(p_shifts['start_total_seconds'], p_shifts['end_total_seconds']))
        
        if not shift_intervals:
            continue

        # Filter Game Events for Goals
        # STRICTLY 5v5 Goals based on Event Metadata
        goals = df_game[
            (df_game['event'].str.lower() == 'goal') & 
            (df_game['game_state'] == '5v5') &
            (df_game['is_net_empty'] == 0) # usually implied by 5v5 but safe to add
        ]
        
        gf = 0
        ga = 0
        
        for idx, row in goals.iterrows():
            t = row['total_time_elapsed_seconds']
            
            is_on_ice = False
            for s, e in shift_intervals:
                if s <= t <= e:
                    is_on_ice = True
                    break
            
            if is_on_ice:
                goal_type = 'GF' if row['team_id'] == team_id else 'GA'
                if goal_type == 'GF':
                    gf += 1
                else:
                    ga += 1
                
                # Collect Data
                collected_goals.append({
                    'game_id': gid,
                    'period': row['period'],
                    'period_time': row['period_time'],
                    'total_time': t,
                    'event': row['event'],
                    'description': row.get('event_description', row.get('secondary_type', '')),
                    'goal_team_id': row['team_id'],
                    'player_team_id': team_id,
                    'type': goal_type,
                    'strength': row.get('strength_state') or row.get('strength_code', 'UNK'),
                    'game_state': row['game_state']
                })

        if gf > 0 or ga > 0:
            print(f"Game {gid}: {gf} GF, {ga} GA (On-Ice 5v5)")
            
        total_gf += gf
        total_ga += ga
        
    print(f"\n--- TOTALS for {player_name} ---")
    print(f"5v5 Goals For: {total_gf}")
    print(f"5v5 Goals Against: {total_ga}")

    # Save CSV
    if collected_goals:
        out_path = f'analysis/players/20252026/sennecke_5v5_goals.csv'
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pd.DataFrame(collected_goals).to_csv(out_path, index=False)
        print(f"Saved events to {out_path}")
    else:
        print("No goals found to save.")

if __name__ == "__main__":
    calc_player_goals()
