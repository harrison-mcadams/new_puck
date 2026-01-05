import pandas as pd
import json
import glob
import os
from puck import nhl_api

def get_clock_sec(period, time_str):
    m, s = map(int, time_str.split(':'))
    return (period - 1) * 1200 + m * 60 + s

def check_game(game_id = 2024020197, goal_event_id = 197):
    print(f"Checking Game {game_id} Goal {goal_event_id}")
    
    # 1. PBP Data
    df_pbp = pd.read_csv('data/20242025/20242025_df.csv')
    df_game = df_pbp[df_pbp['game_id'] == game_id]
    
    print(f"PBP Events for Game: {len(df_game)}")
    
    # Check for blocked shots
    blocks = df_game[df_game['event'] == 'blocked-shot']
    print(f"Blocked Shots: {len(blocks)}")
    if not blocks.empty:
        print(blocks[['period', 'period_time', 'total_time_elapsed_s']].head())
    
    # 2. Tracking Data
    edge_files = glob.glob(f"data/edge_goals/20242025/game_{game_id}_goal_{goal_event_id}_*.json")
    if not edge_files:
        print("No edge files found.")
        return

    edge_json = edge_files[0]
    pos_csv = edge_json.replace('_edge.json', '_positions.csv')
    
    with open(edge_json, 'r') as f:
        meta = json.load(f)
    goal_unix_ms = meta[0]['timeStamp']
    print(f"Edge Goal Unix: {goal_unix_ms}")
    
    # 3. Game Feed for Clock
    feed = nhl_api.get_game_feed(game_id)
    plays = feed.get('plays', [])
    goal_play = next((p for p in plays if str(p.get('eventId')) == str(goal_event_id)), None)
    
    if goal_play:
        period = goal_play.get('periodDescriptor', {}).get('number', 1)
        time_in_period = goal_play.get('timeInPeriod', '00:00')
        goal_clock_sec = get_clock_sec(period, time_in_period)
        print(f"Goal Clock Sec (Feed): {goal_clock_sec}")
    else:
        print("Goal play not found in feed.")
        goal_clock_sec = 0

    # 4. Window Eval
    df_pos = pd.read_csv(pos_csv)
    t_min = df_pos['timestamp'].min()
    t_max = df_pos['timestamp'].max()
    
    w_start = goal_clock_sec + (t_min - goal_unix_ms) / 1000.0
    w_end = goal_clock_sec + (t_max - goal_unix_ms) / 1000.0
    
    print(f"Tracking Window (Clock Sec): {w_start:.2f} to {w_end:.2f}")
    
    # Check overlap
    matches = blocks[
        (blocks['total_time_elapsed_s'] >= w_start - 5.0) & 
        (blocks['total_time_elapsed_s'] <= w_end + 5.0)
    ]
    print(f"Matches in +/- 5s window: {len(matches)}")
    if not matches.empty:
        print(matches[['total_time_elapsed_s', 'x', 'y']])

if __name__ == "__main__":
    check_game()
