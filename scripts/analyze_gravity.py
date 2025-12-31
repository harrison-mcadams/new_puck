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

BASELINE_FILE = os.path.join(r"c:\Users\harri\Desktop\new_puck\data\edge_goals", "mod_baseline.csv")

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
                    'rel_on_puck_mean_dist_ft', 'rel_off_puck_mean_dist_ft',
                    f"Relative On-Puck vs Off-Puck MOD ({season}) - {group_name}",
                    os.path.join(output_dir, f"gravity_rel_corr_{season}_{group_name.lower()}.png")
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
    plt.xlabel("Relative Mean Opponent Distance (ft)\n<-- Higher Relative Pressure | More Relative Space -->", fontsize=12)
    plt.ylabel("Relative Avg Nearest Opponent Distance (ft)\n<-- In Traffic | Finding Open Space -->", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Add zero lines
    plt.axvline(0, color='black', alpha=0.3, linestyle='--')
    plt.axhline(0, color='black', alpha=0.3, linestyle='--')
    
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
    plt.xlabel("Relative On-Puck MOD (ft)", fontsize=11)
    plt.ylabel("Relative Off-Puck MOD (ft)", fontsize=11)
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.grid(True, linestyle=':', alpha=0.5)
    
    # Add zero lines
    plt.axvline(0, color='black', alpha=0.3)
    plt.axhline(0, color='black', alpha=0.3)
    
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()

def analyze_gravity():
    import glob
    
    # Scan CSV files directly from season directories
    csv_files = []
    for season in ['20242025', '20252026']:
        season_dir = os.path.join(DATA_DIR, season)
        if os.path.exists(season_dir):
            for f in glob.glob(os.path.join(season_dir, "*_positions.csv")):
                # Parse filename: game_GAMEID_goal_EVENTID_positions.csv
                basename = os.path.basename(f)
                parts = basename.replace('_positions.csv', '').split('_')
                if len(parts) >= 4 and parts[0] == 'game' and parts[2] == 'goal':
                    csv_files.append({
                        'path': f,
                        'game_id': parts[1],
                        'event_id': parts[3],
                        'season': season
                    })
    
    print(f"Found {len(csv_files)} goal CSV files to analyze.")

    # Load Baseline
    df_baseline = pd.read_csv(BASELINE_FILE)
    baseline_map = df_baseline.set_index(['x_bin', 'y_bin'])['mean'].to_dict()

    results = []
    skipped_preseason = 0
    
    for idx, file_info in enumerate(csv_files):
        if idx % 500 == 0:
            print(f"  Processing {idx}/{len(csv_files)}...")
            
        game_id = file_info['game_id']
        event_id = file_info['event_id']
        season = file_info['season']
        pos_file = file_info['path']
            
        if len(game_id) >= 6 and game_id[4:6] == '01':
            skipped_preseason += 1
            continue
            
        try:
            df_pos = pd.read_csv(pos_file)
        except: continue
            
        if df_pos.empty: continue

        # Standardize units
        if df_pos['x'].abs().max() > 120:
             df_pos['x'] = (df_pos['x'] - 1200.0) / 12.0
             df_pos['y'] = -(df_pos['y'] - 510.0) / 12.0
        
        df_pos['entity_id'] = df_pos['entity_id'].astype(str)
        
        # NORMALIZE TO POSITIVE END
        # Check the mean X position of players in the LAST 50 frames to determine end
        last_frame = df_pos['frame_idx'].max()
        end_data = df_pos[(df_pos['entity_type'] == 'player') & (df_pos['frame_idx'] > last_frame - 50)]
        if end_data.empty: continue
        
        end_mean_x = end_data['x'].mean()
        
        # If play ends at Negative X, flip everything to Positive X
        if end_mean_x < 0:
            df_pos['x'] = -df_pos['x']
            df_pos['y'] = -df_pos['y']

        # Now filter to Goal Moment
        df_pos = filter_data_to_goal_moment(df_pos)
        
        df_players = df_pos[df_pos['entity_type'] == 'player']
        if df_players.empty: continue
        
        # EXTRACT GAME STATE (situationCode) AND SCORING TEAM (eventOwnerTeamId)
        feed = get_game_feed(game_id)
        plays = feed.get('plays', [])
        this_play = next((p for p in plays if str(p.get('eventId')) == event_id), None)
        
        scoring_team_id = None
        if this_play:
            off_team_id = this_play.get('details', {}).get('eventOwnerTeamId')
            if off_team_id:
                off_team_val = float(off_team_id)
                teams = df_players['team_id'].unique()
                
                # Match API ID to Tracking ID
                if off_team_val in teams:
                    scoring_team_id = off_team_val
                else:
                    # Try fuzzy matching int values
                    match = next((t for t in teams if int(t) == int(off_team_val)), None)
                    if match:
                        scoring_team_id = match
        
        if scoring_team_id is None:
             # Fallback if API fails? Skip to be safe
             continue
        
        game_state = 'UNK'
        if this_play and 'situationCode' in this_play:
            sc = this_play['situationCode'] # e.g. "1551"
            if len(sc) == 4:
                if sc[1] == '5' and sc[2] == '5':
                    game_state = '5v5'
                else:
                    game_state = f"{sc[2]}v{sc[1]}"
        
        # FILTER: Only 5v5
        if game_state != '5v5':
            continue

        roster = get_roster_map(game_id)
        
        # df_players already defined above for team detection
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
                    
                    # Expected MOD for this location
                    xb = (mx // 5) * 5
                    yb = (my // 5) * 5
                    expected_mod = baseline_map.get((xb, yb), np.nan)
                    
                    states[state].append((dists.mean(), dists.min(), expected_mod))
                except: pass
            
            if not states['on_puck'] and not states['off_puck']: continue
            
            def _avg(lst, idx):
                # Filter out nans for the expected_mod column (index 2)
                clean = [x[idx] for x in lst if not np.isnan(x[idx])]
                if not clean: return np.nan
                return np.mean(clean)

            on_puck_mod = _avg(states['on_puck'], 0)
            off_puck_mod = _avg(states['off_puck'], 0)
            exp_on_puck = _avg(states['on_puck'], 2)
            exp_off_puck = _avg(states['off_puck'], 2)

            res = {
                'season': season, 'game_id': game_id, 'event_id': event_id,
                'player_id': pid, 'role': 'Teammate',  # Scorer ID not available in file-based approach
                'game_state': game_state,
                'on_puck_mean_dist_ft': on_puck_mod,
                'on_puck_nearest_dist_ft': _avg(states['on_puck'], 1),
                'off_puck_mean_dist_ft': off_puck_mod,
                'off_puck_nearest_dist_ft': _avg(states['off_puck'], 1),
                'rel_on_puck_mean_dist_ft': on_puck_mod - exp_on_puck,
                'rel_off_puck_mean_dist_ft': off_puck_mod - exp_off_puck,
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
        'rel_on_puck_mean_dist_ft': 'mean',
        'rel_off_puck_mean_dist_ft': 'mean',
        'event_id': 'count'
    }).rename(columns={'event_id': 'goals_on_ice_count'}).reset_index()
    
    df_agg.to_csv(os.path.join(agg_dir, "player_gravity_season.csv"), index=False)
    visualize_gravity(df_agg, agg_dir)

if __name__ == "__main__":
    analyze_gravity()
