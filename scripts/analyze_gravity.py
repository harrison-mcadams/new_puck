import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import logging
from puck.edge import transform_coordinates, filter_data_to_goal_moment
from puck.nhl_api import get_game_feed
from puck.possession import infer_possession_events
import matplotlib.pyplot as plt

DATA_DIR = r"c:\Users\harri\Desktop\new_puck\data\edge_goals"
METADATA_FILE = os.path.join(DATA_DIR, "metadata.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "gravity_analysis.csv")

def get_roster_map(game_id):
    try:
        feed = get_game_feed(game_id)
        if not feed: return {}
        
        roster = {}
        player_lists = []
        
        # 1. Top Level 'rosterSpots' (Common in new API)
        if 'rosterSpots' in feed:
            player_lists.extend(feed['rosterSpots'])
            
        # 2. Nested 'playerByTeam'
        if 'playerByTeam' in feed:
            for side in ['homeTeam', 'awayTeam']:
                pbt = feed['playerByTeam'].get(side, {})
                for key in ['roster', 'rosterSpots', 'players']:
                    if key in pbt:
                        player_lists.extend(pbt[key])
                        
        # 3. Old API 'gameData'
        if 'gameData' in feed and 'players' in feed['gameData']:
            for pid, pdata in feed['gameData']['players'].items():
                player_lists.append(pdata)
                
        for p in player_lists:
            pid = p.get('player_id') or p.get('id') or p.get('playerId')
            if not pid: continue
            
            name = f"Player {pid}" # Default
            
            # Name Resolution Strategy
            if 'firstName' in p and 'lastName' in p:
                fname = p['firstName']
                lname = p['lastName']
                if isinstance(fname, dict): fname = fname.get('default', str(fname))
                if isinstance(lname, dict): lname = lname.get('default', str(lname))
                name = f"{fname} {lname}"
            elif 'fullName' in p:
                name = p['fullName']
            elif 'name' in p:
                name_obj = p.get('name')
                if isinstance(name_obj, dict):
                    name = name_obj.get('default', str(name_obj))
                else:
                    name = str(name_obj)
            
            clean_name = name.strip()
            
            # Get Position
            pos_code = p.get('positionCode')
            if not pos_code and 'position' in p:
                if isinstance(p['position'], dict):
                    pos_code = p['position'].get('code')
                else:
                    pos_code = p['position']
            if not pos_code: pos_code = 'UNK'

            roster[str(pid)] = {'name': clean_name, 'position': pos_code}
            try:
                roster[int(pid)] = {'name': clean_name, 'position': pos_code}
            except:
                pass
            
        return roster
    except Exception as e:
        return {}

def visualize_gravity(df_agg, output_dir):
    try:
        # Ensure numeric columns
        cols = [
            'on_puck_mean_dist_ft', 'on_puck_nearest_dist_ft',
            'off_puck_mean_dist_ft', 'off_puck_nearest_dist_ft',
            'goals_on_ice_count'
        ]
        for col in cols:
            df_agg[col] = pd.to_numeric(df_agg[col], errors='coerce')
        
        # Filter Goalies
        df_agg = df_agg[df_agg['position'] != 'G']
        
        seasons = df_agg['season'].unique()
        print(f"\nGenerating plots for seasons: {seasons}")
        
        groups = {
            'Forwards': ['C', 'L', 'R', 'LW', 'RW'],
            'Defense': ['D', 'LD', 'RD']
        }
        
        for season in seasons:
            df_season = df_agg[df_agg['season'] == season]
            if df_season.empty: continue
            
            for group_name, positions in groups.items():
                df_group = df_season[df_season['position'].isin(positions)].copy()
                if df_group.empty: continue

                # 1. On-Puck Plot
                _create_scatter(
                    df_group, 
                    'on_puck_mean_dist_ft', 'on_puck_nearest_dist_ft',
                    f"On-Puck Gravity ({season}) - {group_name}",
                    os.path.join(output_dir, f"gravity_plot_{season}_{group_name.lower()}_on_puck.png"),
                    'coral' if group_name == 'Forwards' else 'cornflowerblue'
                )

                # 2. Off-Puck Plot
                _create_scatter(
                    df_group, 
                    'off_puck_mean_dist_ft', 'off_puck_nearest_dist_ft',
                    f"Off-Puck (Threat) Gravity ({season}) - {group_name}",
                    os.path.join(output_dir, f"gravity_plot_{season}_{group_name.lower()}_off_puck.png"),
                    'orange' if group_name == 'Forwards' else 'royalblue'
                )

                # 3. Correlation Plot
                _create_correlation(
                    df_group,
                    'on_puck_mean_dist_ft', 'off_puck_mean_dist_ft',
                    f"On-Puck vs Off-Puck MOD ({season}) - {group_name}",
                    os.path.join(output_dir, f"gravity_corr_{season}_{group_name.lower()}.png")
                )
            
    except Exception as e:
        print(f"Visualization error: {e}")

def _create_scatter(df, x_col, y_col, title, out_path, color):
    # Filter valid rows
    df_valid = df.dropna(subset=[x_col, y_col])
    if df_valid.empty: return

    plt.figure(figsize=(12, 8))
    sizes = df_valid['goals_on_ice_count'] * 30
    sizes = sizes.clip(lower=20, upper=300)
    
    plt.scatter(
        df_valid[x_col], df_valid[y_col], 
        s=sizes, alpha=0.6, edgecolors='w', linewidth=0.5, color=color
    )
    
    for _, row in df_valid.iterrows():
        lbl = str(row['player_name'])
        if len(lbl) > 15: lbl = lbl.split(' ')[-1]
        plt.annotate(
            lbl, (row[x_col], row[y_col]),
            fontsize=9, alpha=0.9, xytext=(3, 3), textcoords='offset points'
        )
    
    plt.title(title, fontsize=16)
    plt.xlabel("Mean Opponent Distance (ft)\n<-- Higher Pressure | More Space -->", fontsize=12)
    plt.ylabel("Avg Nearest Opponent Distance (ft)\n<-- In Traffic | Finding Open Space -->", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {out_path}")

def _create_correlation(df, x_col, y_col, title, out_path):
    df_valid = df.dropna(subset=[x_col, y_col])
    if df_valid.empty: return

    plt.figure(figsize=(10, 10))
    plt.scatter(df_valid[x_col], df_valid[y_col], alpha=0.7, color='purple')
    
    # Add diagonal line
    mi = min(df_valid[x_col].min(), df_valid[y_col].min())
    ma = max(df_valid[x_col].max(), df_valid[y_col].max())
    plt.plot([mi, ma], [mi, ma], 'k--', alpha=0.2)
    
    for _, row in df_valid.iterrows():
        lbl = row['player_name'].split(' ')[-1]
        plt.annotate(lbl, (row[x_col], row[y_col]), fontsize=8)
        
    plt.title(title, fontsize=14)
    plt.xlabel("On-Puck Mean Opponent Distance (ft)", fontsize=11)
    plt.ylabel("Off-Puck Mean Opponent Distance (ft)", fontsize=11)
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()

def analyze_gravity():
    if not os.path.exists(METADATA_FILE):
        print(f"Metadata file not found: {METADATA_FILE}")
        return

    df_meta = pd.read_csv(METADATA_FILE)
    print(f"Found {len(df_meta)} goals to analyze.")

    results = []
    skipped_preseason = 0
    
    for idx, row in df_meta.iterrows():
        try:
            game_id = str(int(row['game_id']))
            event_id = str(int(row['event_id']))
            scorer_id = str(int(row['scorer_id']))
            season = str(row['season'])
        except:
            continue
            
        if len(game_id) >= 6 and game_id[4:6] == '01':
            skipped_preseason += 1
            continue
            
        pos_file = os.path.join(DATA_DIR, f"game_{game_id}_goal_{event_id}_positions.csv")
        if not os.path.exists(pos_file): continue
            
        try:
            df_pos = pd.read_csv(pos_file)
        except: continue
            
        if df_pos.empty: continue

        # Standardize units
        if df_pos['x'].abs().max() > 120:
             df_pos['x'] = (df_pos['x'] - 1200.0) / 12.0
             df_pos['y'] = -(df_pos['y'] - 510.0) / 12.0
        
        df_pos['entity_id'] = df_pos['entity_id'].astype(str)
        df_pos = filter_data_to_goal_moment(df_pos)
        
        scorer_frames = df_pos[df_pos['entity_id'] == scorer_id]
        if scorer_frames.empty: continue
        
        scoring_team_id = scorer_frames.iloc[0]['team_id']
        
        # EXTRACT GAME STATE (situationCode)
        feed = get_game_feed(game_id)
        plays = feed.get('plays', [])
        this_play = next((p for p in plays if str(p.get('eventId')) == event_id), None)
        
        game_state = 'UNK'
        if this_play and 'situationCode' in this_play:
            sc = this_play['situationCode'] # e.g. "1551"
            # format: [awayGoalie, awaySkaters, homeSkaters, homeGoalie]
            # We want to know if it's 5v5.
            # But wait, we need to know WHICH team is home/away to label correctly?
            # Actually for gravity, we just care if it's 5v5 skaters.
            if len(sc) == 4:
                if sc[1] == '5' and sc[2] == '5':
                    game_state = '5v5'
                else:
                    game_state = f"{sc[2]}v{sc[1]}"
        
        # FILTER: Only 5v5
        if game_state != '5v5':
            continue

        roster = get_roster_map(game_id)
        
        df_players = df_pos[df_pos['entity_type'] == 'player']
        off_players = df_players[df_players['team_id'] == scoring_team_id]['entity_id'].unique()
        def_players_df = df_players[df_players['team_id'] != scoring_team_id]
        
        if def_players_df.empty: continue

        # Filter out Goalie from Defenders
        def_pids = def_players_df['entity_id'].unique()
        goalie_ids = [pid for pid in def_pids if (roster.get(str(pid)) or {}).get('position') == 'G' or (roster.get(int(pid)) or {}).get('position') == 'G']
        if goalie_ids:
            def_players_df = def_players_df[~def_players_df['entity_id'].isin(goalie_ids)]
            if def_players_df.empty: continue

        # Possession logic
        poss_events = infer_possession_events(df_pos, threshold_ft=6.0)
        poss_player_map = {}
        if not poss_events.empty:
            for _, pev in poss_events.iterrows():
                if pev['is_possession']:
                    for f in range(pev['start_frame'], pev['end_frame'] + 1):
                        poss_player_map[f] = str(pev['player_id'])

        for pid in off_players:
            p_track = df_players[df_players['entity_id'] == pid].set_index('frame_idx')
            common_frames = p_track.index.intersection(def_players_df['frame_idx'].unique())
            
            # metrics
            states = {'on_puck': [], 'off_puck': []}
            
            for frame in common_frames:
                poss_pid = poss_player_map.get(frame)
                if not poss_pid: continue # Team doesn't have it
                
                # Verify teammate has it? 
                # poss_pid must be in our team or we have to trust the map if we filter'd correctly.
                # Let's check team of poss_pid
                # ... checking team is hard without full roster, but we know our team.
                
                # Simple logic: If anyone has it, and WE are on offense, it's offensive possession.
                
                state = 'on_puck' if poss_pid == pid else 'off_puck'
                
                try:
                    me = p_track.loc[frame]
                    if isinstance(me, pd.DataFrame): me = me.iloc[0]
                    mx, my = me['x'], me['y']
                    
                    def_frame = def_players_df[def_players_df['frame_idx'] == frame]
                    if def_frame.empty: continue
                    
                    dx = def_frame['x'] - mx
                    dy = def_frame['y'] - my
                    dists = np.sqrt(dx**2 + dy**2)
                    
                    states[state].append((dists.mean(), dists.min()))
                except: pass
            
            if not states['on_puck'] and not states['off_puck']: continue
            
            def _avg(lst, idx):
                if not lst: return np.nan
                return np.mean([x[idx] for x in lst])

            res = {
                'season': season, 'game_id': game_id, 'event_id': event_id,
                'player_id': pid, 'role': 'Scorer' if pid == scorer_id else 'Teammate',
                'game_state': game_state,
                'on_puck_mean_dist_ft': _avg(states['on_puck'], 0),
                'on_puck_nearest_dist_ft': _avg(states['on_puck'], 1),
                'off_puck_mean_dist_ft': _avg(states['off_puck'], 0),
                'off_puck_nearest_dist_ft': _avg(states['off_puck'], 1),
                'on_puck_frames': len(states['on_puck']),
                'off_puck_frames': len(states['off_puck'])
            }
            
            # Add name/pos
            p_data = roster.get(str(pid)) or roster.get(int(pid)) or {}
            res['player_name'] = p_data.get('name', f"Player {pid}")
            res['position'] = p_data.get('position', 'UNK')
            
            results.append(res)
            
    print(f"Skipped {skipped_preseason} pre-season games.")
    df_results = pd.DataFrame(results)
    if df_results.empty: return

    df_results.to_csv(OUTPUT_FILE, index=False)
    
    print("\n[Aggregating Results by Season]")
    agg_dir = os.path.join("analysis", "gravity")
    os.makedirs(agg_dir, exist_ok=True)
    
    df_agg = df_results.groupby(['season', 'player_id']).agg({
        'player_name': 'first',
        'position': 'first',
        'on_puck_mean_dist_ft': 'mean',
        'on_puck_nearest_dist_ft': 'mean',
        'off_puck_mean_dist_ft': 'mean',
        'off_puck_nearest_dist_ft': 'mean',
        'event_id': 'count'
    }).rename(columns={'event_id': 'goals_on_ice_count'}).reset_index()
    
    df_agg.to_csv(os.path.join(agg_dir, "player_gravity_season.csv"), index=False)
    visualize_gravity(df_agg, agg_dir)

if __name__ == "__main__":
    analyze_gravity()
