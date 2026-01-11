
"""
Script to list sample shot attempts exactly at (69, 22).
"""
import sys
import os
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from puck import fit_xgs

def main():
    print("Loading Data...")
    try:
        df = fit_xgs.load_all_seasons_data()
    except:
        df = fit_xgs.load_data()
        
    # Filter for Exact Match at (69, 22)
    # We want "Shot Attempts": Goal, Shot, Miss, Block
    events = ['goal', 'shot-on-goal', 'missed-shot', 'blocked-shot']
    
    mask = (df['x'] == 69.0) & (df['y'] == 22.0) & (df['event'].isin(events))
    subset = df[mask]
    
    print(f"\nFound {len(subset)} events exactly at (69, 22).")
    print("Showing first 20 examples:\n")
    

    # Define possible columns map
    col_map = {
        'season': ['season', 'Season'],
        'game_id': ['game_id', 'Game_Id', 'gameId'],
        'date': ['game_date', 'Date', 'date'],
        'period': ['period', 'Period'],
        'time': ['period_seconds', 'period_time', 'time_in_period', 'Time'],
        'event': ['event', 'event_type', 'Event'],
        'team': ['team_name', 'event_team', 'Team'],
        'player': ['shooter_name', 'event_player_1', 'Player_Name', 'p1_name'],
        'x': ['x', 'X_Coordinate'],
        'y': ['y', 'Y_Coordinate'],
        'desc': ['description', 'Description']
    }

    final_cols = []
    
    # Select available columns
    for label, options in col_map.items():
        found = False
        for opt in options:
            if opt in df.columns:
                final_cols.append(opt)
                found = True
                break
    
    # Display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 100)
    
    if len(final_cols) == 0:
        print("No expected columns found. Available columns:")
        print(df.columns.tolist())
    else:
        print(subset[final_cols].head(20).to_string(index=False))

if __name__ == "__main__":
    main()
