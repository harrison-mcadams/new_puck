
import os
import glob
import pandas as pd
import sys

# Setup paths
DATA_DIR = r"c:\Users\harri\Desktop\new_puck\data\edge_goals"
OUTPUT_FILE = os.path.join(DATA_DIR, "gravity_analysis.csv")

# 1. SCAN FILES like analyze_gravity.py
csv_files = []
print("Scanning files...")
for season in ['20242025', '20252026']:
    season_dir = os.path.join(DATA_DIR, season)
    if os.path.exists(season_dir):
        files = glob.glob(os.path.join(season_dir, "*_positions.csv"))
        print(f"Season {season}: Found {len(files)} files.")
        for f in files:
            basename = os.path.basename(f)
            if '2024020462' in basename:
                print(f"SAW FILE: {basename}")
            
            parts = basename.replace('_positions.csv', '').split('_')
            # Expect: game_2024020462_goal_691_positions.csv
            # parts: ['game', '2024020462', 'goal', '691']
            if len(parts) >= 4 and parts[0] == 'game' and parts[2] == 'goal':
                game_id = parts[1]
                if '2024020462' in basename:
                   print(f"  PARSED: Game={game_id} Event={parts[3]}")
                
                # Filter strictly for one PHI game we saw earlier
                csv_files.append({'path': f, 'game_id': game_id, 'event_id': parts[3], 'season': season})

print(f"Total potential files: {len(csv_files)}")

# 2. CHECK PROCESSED KEYS
processed_keys = set()
if os.path.exists(OUTPUT_FILE):
    df_existing = pd.read_csv(OUTPUT_FILE)
    if not df_existing.empty:
        for _, row in df_existing.iterrows():
            processed_keys.add((str(row['game_id']), str(row['event_id'])))
    print(f"Loaded {len(processed_keys)} processed keys.")

# 3. IDENTIFY TARGET PHI GAME
# Using a game ID from the 'ls' output: game_2024020462_goal_691_positions.csv
target_game = '2024020462'
target_event = '691'

target_file = None
for f in csv_files:
    if f['game_id'] == target_game and f['event_id'] == target_event:
        target_file = f
        break

if not target_file:
    print(f"CRITICAL: Target PHI file {target_game} goal {target_event} NOT FOUND in scan.")
    # Print potential partial matches
    cols = [f for f in csv_files if f['game_id'] == target_game]
    print(f"Files for game {target_game}: {len(cols)}")
else:
    print(f"Found target file: {target_file['path']}")
    is_processed = (str(target_game), str(target_game)) in processed_keys
    print(f"Is already processed? {is_processed}")
    
    # 4. TRACE PROCESS
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from puck.nhl_api import get_game_feed, get_roster_map
    from puck.edge import filter_data_to_goal_moment
    from puck.possession import infer_possession_events
    import numpy as np

    print("\n--- TRACING PROCESS FOR TARGET FILE ---")
    game_id = target_file['game_id']
    event_id = target_file['event_id']
    season = target_file['season']
    pos_file = target_file['path']

    # 1. API
    print("Fetching Feed...")
    feed = get_game_feed(game_id)
    if not feed:
        print("FAIL: Feed not found.")
        sys.exit()
    
    print("Fetching Roster...")
    roster = get_roster_map(game_id)
    if not roster:
        print("WARN: Roster empty.")

    # 2. LOAD DATA
    print(f"Loading CSV: {pos_file}")
    df_pos = pd.read_csv(pos_file)
    if df_pos.empty:
        print("FAIL: CSV empty.")
        sys.exit()
    print(f"CSV Rows: {len(df_pos)}")

    # 3. PLAY INFO
    plays = feed.get('plays', [])
    play = next((p for p in plays if str(p.get('eventId')) == event_id), None)
    if not play:
        print(f"FAIL: Event {event_id} not found in feed.")
        print(f"Available events: {[p.get('eventId') for p in plays[:5]]}...")
        sys.exit()
    
    print("Play Found.")
    off_team_id = play.get('details', {}).get('eventOwnerTeamId')
    print(f"Event Owner Team ID: {off_team_id}")
    
    scoring_team_id = float(off_team_id) if off_team_id else None
    
    sc = play.get('situationCode', '')
    game_state = '5v5' if (len(sc) == 4 and sc[1] == '5' and sc[2] == '5') else 'OTHER'
    print(f"Situation Code: {sc} -> Game State: {game_state}")
    
    if game_state != '5v5':
        print("FAIL: Not 5v5.")
        # sys.exit() # Keep going to see info

    # 4. PLAYERS
    df_pos['entity_id'] = df_pos['entity_id'].astype(str)
    # Norm logic skipped for brevity, assuming it doesn't empty the DF
    
    df_players = df_pos[df_pos['entity_type'] == 'player']
    off_pids = df_players[df_players['team_id'] == scoring_team_id]['entity_id'].unique()
    print(f"Scoring Team Players found (ID {scoring_team_id}): {len(off_pids)}")
    print(f"Player IDs: {off_pids}")
    
    if len(off_pids) == 0:
        print("FAIL: No players found for scoring team.")
        print(f"Available Team IDs in CSV: {df_players['team_id'].unique()}")

    # Check Roster Mapping
    for pid in off_pids:
        p_info = roster.get(str(pid))
        print(f"Player {pid}: {p_info}")

