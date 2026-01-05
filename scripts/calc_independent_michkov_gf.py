import pandas as pd
import json
import os
import glob
import re

# Constants
TARGET_PID = 8484387 # Matvei Michkov
TARGET_TEAM_ID = 4   # Philadelphia Flyers
RAW_FEED_CSV = r'data/20252026_raw_game_feeds.csv'
TRACKING_DIR = r'data/edge_goals/20252026'

def get_goal_team_map():
    print("Building goal team map...")
    goal_team_map = {} # (game_id, event_id) -> team_id
    
    try:
        # Read in chunks to avoid memory issues if large
        chunksize = 100
        for chunk in pd.read_csv(RAW_FEED_CSV, chunksize=chunksize):
            for _, row in chunk.iterrows():
                try:
                    game_id = row['game_id']
                    feed = json.loads(row['feed'])
                    
                    if 'plays' in feed:
                         for p in feed['plays']:
                             if p.get('typeDescKey') == 'goal':
                                 event_id = p.get('eventId')
                                 # eventOwnerTeamId is usually the scoring team
                                 team_id = p.get('details', {}).get('eventOwnerTeamId')
                                 # Debug specific game
                                 if game_id == 2025020008:
                                     print(f"DEBUG_MAP: Found Goal in Game {game_id}: Event {event_id} Team {team_id}")
                                 
                                 if event_id and team_id:
                                     goal_team_map[(game_id, event_id)] = team_id
                except Exception as e:
                    continue
    except Exception as e:
        print(f"Error reading feeds: {e}")
        
    print(f"Found {len(goal_team_map)} goals in raw feeds.")
    team4_goals = [k for k, v in goal_team_map.items() if v == TARGET_TEAM_ID]
    print(f"Goals by Team {TARGET_TEAM_ID}: {len(team4_goals)}")
    
    return goal_team_map

def check_tracking_file(f_path):
    try:
        df = pd.read_csv(f_path)
        if df.empty: return False, False
        
        frame0 = df[df['frame_idx'] == df['frame_idx'].min()]
        players = frame0[frame0['entity_type'] == 'player']
        team_counts = players['team_id'].value_counts()
        
        is_5v5 = False
        if len(team_counts) == 2:
            c1 = team_counts.iloc[0]
            c2 = team_counts.iloc[1]
            if c1 == 6 and c2 == 6:
                is_5v5 = True
        
        has_michkov = TARGET_PID in players['entity_id'].values
        
        return is_5v5, has_michkov, team_counts.to_dict()

    except Exception as e:
        return False, False, {}

def main():
    goals_map = get_goal_team_map()
    
    files = glob.glob(os.path.join(TRACKING_DIR, "game_2025*_positions.csv"))
    print(f"Found {len(files)} tracking files.")
    
    michkov_goals = []
    
    with open('debug_output.txt', 'w') as log:
        for f in files:
            basename = os.path.basename(f)
            match = re.search(r'game_(\d+)_goal_(\d+)_positions.csv', basename)
            if match:
                game_id = int(match.group(1))
                event_id = int(match.group(2))
                
                scoring_team = goals_map.get((game_id, event_id))
                
                if scoring_team != TARGET_TEAM_ID:
                    continue
                
                # It is a goal by Team 4!
                is_5v5, has_michkov, counts = check_tracking_file(f)
                
                log.write(f"checking {basename}: 5v5={is_5v5}, Michkov={has_michkov}, Counts={counts}\n")
                
                if is_5v5 and has_michkov:
                    michkov_goals.append((game_id, event_id))

    print(f"\nIndependent Calculation Results:")
    print(f"Matvei Michkov (8484387) 5v5 On-Ice Goals: {len(michkov_goals)}")
    with open('debug_output.txt', 'a') as log:
        log.write(f"Total: {len(michkov_goals)}")

if __name__ == "__main__":
    main()
