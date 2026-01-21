
import pandas as pd
import numpy as np
import os
import sys

# Setup path
sys.path.append(os.getcwd())
from puck import plot
from puck import timing

def debug_orientation():
    # Load a known game where we can check Home/Away
    # Let's find a game for a specific team, e.g., SEA
    # We'll use the season dataframe
    
    print("Loading season data...")
    df = timing.load_season_df('20252026')
    
    # Pick a game where SEA is AWAY
    sea_away_games = df[df['away_abb'] == 'SEA']['game_id'].unique()
    if len(sea_away_games) == 0:
        print("No SEA away games found.")
        return

    game_id = sea_away_games[0]
    print(f"Checking Game {game_id} (SEA is Away)...")
    
    df_game = df[df['game_id'] == game_id].copy()
    
    sea_id = df_game.iloc[0]['away_id']
    opp_id = df_game.iloc[0]['home_id']
    print(f"SEA ID: {sea_id}, OPP ID: {opp_id}")

    # Check RAW x_adj
    if 'x_adj' in df_game.columns:
        print("\n--- RAW x_adj check ---")
        sea_raw = df_game[(df_game['team_id'] == sea_id) & (df_game['event'].isin(['shot-on-goal', 'missed-shot', 'goal']))]
        opp_raw = df_game[(df_game['team_id'] == opp_id) & (df_game['event'].isin(['shot-on-goal', 'missed-shot', 'goal']))]
        print(f"SEA (Away) Raw x_adj Mean: {sea_raw['x_adj'].mean():.2f}")
        print(f"OPP (Home) Raw x_adj Mean: {opp_raw['x_adj'].mean():.2f}")
    
    
    # Simulate what xgs_map does
    # It calls adjust_xy_for_homeaway with split_mode='team_not_team' and team_for_heatmap='SEA'
    
    print("Applying adjust_xy_for_homeaway(mode='team_not_team', team='SEA')...")
    df_adj = plot.adjust_xy_for_homeaway(df_game, split_mode='team_not_team', team_for_heatmap='SEA')
    
    
    sea_events = df_adj[df_adj['team_id'] == sea_id]
    opp_events = df_adj[df_adj['team_id'] == opp_id]
    
    # Filter to shots only
    shots = ['shot-on-goal', 'missed-shot', 'goal']
    sea_shots = sea_events[sea_events['event'].isin(shots)]
    opp_shots = opp_events[opp_events['event'].isin(shots)]
    
    # Check X coordinates (x_a)
    # SEA (Team) should be LEFT (negative x_a)
    # OPP (Other) should be RIGHT (positive x_a)
    
    print("\n--- SEA (Team) Stats ---")
    if 'x_a' in sea_shots.columns:
        mean_x = sea_shots['x_a'].mean()
        print(f"Mean x_a: {mean_x:.2f}")
        print(f"Count < 0 (Left): {(sea_shots['x_a'] < 0).sum()} / {len(sea_shots)}")
        print(f"Count > 0 (Right): {(sea_shots['x_a'] > 0).sum()} / {len(sea_shots)}")
    else:
        print("Column x_a not found!")

    print("\n--- OPP (Other) Stats ---")
    if 'x_a' in opp_shots.columns:
        mean_x = opp_shots['x_a'].mean()
        print(f"Mean x_a: {mean_x:.2f}")
        print(f"Count < 0 (Left): {(opp_shots['x_a'] < 0).sum()} / {len(opp_shots)}")
        print(f"Count > 0 (Right): {(opp_shots['x_a'] > 0).sum()} / {len(opp_shots)}")
        
    # Check Period 2 behavior specifically if possible (requires period info)
    if 'period' in df_game.columns:
        print("\n--- Period 2 Check ---")
        sea_p2 = sea_shots[sea_shots['period'] == 2]
        if not sea_p2.empty:
            print(f"SEA P2 Mean x_a: {sea_p2['x_a'].mean():.2f}")
        else:
            print("No SEA P2 shots.")

if __name__ == "__main__":
    debug_orientation()
