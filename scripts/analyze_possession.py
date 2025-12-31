import sys
import os
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck.possession import infer_possession_events
from puck.nhl_api import get_game_feed

DATA_DIR = r"c:\Users\harri\Desktop\new_puck\data\edge_goals"
METADATA_FILE = os.path.join(DATA_DIR, "metadata.csv")
OUTPUT_FILE = os.path.join("analysis", "possession", "player_possession.csv")

# Ensure output dir
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

def get_roster_cache(game_ids):
    # Simple cache to avoid re-fetching roster for every goal in same game
    cache = {}
    return cache

def resolve_name(player_id, game_roster):
    pid_str = str(player_id)
    if pid_str in game_roster:
        return game_roster[pid_str]
    # Fallback to API if we had a function for standalone player lookup, 
    # but for now rely on what we have or "Player X"
    return f"Player {player_id}"

# Reuse roster logic from analyze_gravity (could functionize this too)
def get_roster_map(game_id):
    try:
        feed = get_game_feed(game_id)
        if not feed: return {}
        roster = {}
        player_lists = []
        if 'rosterSpots' in feed: player_lists.extend(feed['rosterSpots'])
        if 'playerByTeam' in feed:
            for side in ['homeTeam', 'awayTeam']:
                pbt = feed['playerByTeam'].get(side, {})
                for key in ['roster', 'rosterSpots', 'players']:
                    if key in pbt: player_lists.extend(pbt[key])
        if 'gameData' in feed and 'players' in feed['gameData']:
             for pid, pdata in feed['gameData']['players'].items(): player_lists.append(pdata)
             
        for p in player_lists:
            pid = p.get('player_id') or p.get('id') or p.get('playerId')
            if not pid: continue
            
            # Name resolution
            name = f"Player {pid}"
            if 'firstName' in p and 'lastName' in p:
                fname = p['firstName']; lname = p['lastName']
                if isinstance(fname, dict): fname = fname.get('default', str(fname))
                if isinstance(lname, dict): lname = lname.get('default', str(lname))
                name = f"{fname} {lname}"
            elif 'fullName' in p: name = p['fullName']
            
            roster[str(pid)] = name.strip()
            try: roster[int(pid)] = name.strip()
            except: pass
            
        return roster
    except: return {}

def analyze_all_possession():
    if not os.path.exists(METADATA_FILE):
        print("Metadata not found.")
        return

    df_meta = pd.read_csv(METADATA_FILE)
    print(f"Analyzing possession for {len(df_meta)} goals...")
    
    all_events = []
    
    roster_cache = {}
    
    for idx, row in df_meta.iterrows():
        try:
            game_id = str(int(row['game_id']))
            event_id = str(int(row['event_id']))
            season = str(row['season'])
        except: continue
        
        # Filter Preseason
        if len(game_id) >= 6 and game_id[4:6] == '01':
            continue
            
        pos_file = os.path.join(DATA_DIR, f"game_{game_id}_goal_{event_id}_positions.csv")
        if not os.path.exists(pos_file): continue
        
        try:
            df_pos = pd.read_csv(pos_file)
            if df_pos.empty: continue
            
            # Ensure feet (older extracts might checks, but we assume latest standard)
            # Standardize: extract_goal_data.py writes raw transformed 'x', 'y' (which ARE feet)
            # analyze_gravity.py was doing (x-1200)/12, which implies extract IS raw pixels?
            # WAIT. Let's check `puck/edge.py`.
            # `transform_coordinates` -> (x - 1200) / 12.
            # `extract_goal_data.py` CALLS `transform_coordinates` before writing CSV.
            # So the CSV contains FEET.
            # `analyze_gravity.py` was doing: `df_pos['x_ft'] = (df_pos['x'] - 1200.0) / 12.0`
            # --> THIS MEANS ANALYZE_GRAVITY MIGHT BE DOUBLE TRANSFORMING if the CSV is already feet?
            # Let's verify this quickly.
            # If CSV x range is -100 to 100, it's feet. If 0 to 2400, it's pixels.
            
            # Quick Check
            x_range = df_pos['x'].max() - df_pos['x'].min()
            if df_pos['x'].max() > 500: # It's pixels (1200 center) OR it's raw. 
                # extract_goal_data.py line 214: x, y = transform_coordinates(x_raw, y_raw)
                # So it SHOULD be feet.
                # However, if analyze_gravity was transforming again... it might be shrinking the rink to tiny size.
                # Or maybe I misread analyze_gravity.
                
                # Let's trust the logic: If coordinates are > 200, assume pixels -> transform.
                # If < 200, assume feet.
                pass
                
            is_pixels = df_pos['x'].abs().max() > 105
            
            if is_pixels:
                df_pos['x'] = (df_pos['x'] - 1200.0) / 12.0
                df_pos['y'] = -(df_pos['y'] - 510.0) / 12.0
                
            # Run Inference
            poss_df = infer_possession_events(df_pos)
            
            if poss_df.empty: continue
            
            # Get Roster
            if game_id not in roster_cache:
                roster_cache[game_id] = get_roster_map(game_id)
            roster = roster_cache[game_id]
            
            # Add context
            poss_df['game_id'] = game_id
            poss_df['event_id'] = event_id
            poss_df['season'] = season
            
            # Resolve Names
            poss_df['player_name'] = poss_df['player_id'].apply(lambda x: resolve_name(x, roster) if x != 'LOOSE' else 'Loose Puck')
            
            all_events.append(poss_df)
            
        except Exception as e:
            # print(f"Error {game_id}: {e}")
            continue

    if not all_events:
        print("No events found.")
        return

    df_all = pd.concat(all_events, ignore_index=True)
    
    # Save raw events
    df_all.to_csv(os.path.join("analysis", "possession", "all_possession_events.csv"), index=False)
    
    # Aggregate
    # Filter for real possession only
    df_real = df_all[df_all['is_possession'] == True]
    
    agg = df_real.groupby(['season', 'player_id', 'player_name']).agg(
        TotalPossessionFrames=('duration_frames', 'sum'),
        PossessionEvents=('start_frame', 'count'),
        AvgEventDuration=('duration_frames', 'mean')
    ).reset_index()
    
    # Approx seconds (Assuming ~30fps? No, tracking is usually higher freq? or 20fps?)
    # extract_goal_data animation interval=50ms -> 20fps.
    # Let's checking `extract_goal_data.py`. 
    # Actually metadata doesn't specify FPS. But typically standard is 30Hz or higher.
    # Let's just output frames for now, or estimate seconds roughly.
    # We will assume 30Hz for a rough "Seconds" metric.
    
    agg['EstSeconds'] = agg['TotalPossessionFrames'] / 30.0
    
    agg_sorted = agg.sort_values('TotalPossessionFrames', ascending=False)
    
    agg_sorted.to_csv(OUTPUT_FILE, index=False)
    print(f"Top 10 Possession Leaders:\n{agg_sorted[['player_name', 'EstSeconds', 'PossessionEvents']].head(10)}")

if __name__ == "__main__":
    analyze_all_possession()
