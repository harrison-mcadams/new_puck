import analyze
import timing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json

def run_analysis():
    season = '20252026'
    out_dir_base = 'static/players'
    league_out_dir = os.path.join(out_dir_base, f'{season}/league')
    os.makedirs(league_out_dir, exist_ok=True)
    
    print(f"Loading season data for {season}...")
    df_data = timing.load_season_df(season)
    if df_data is None or df_data.empty:
        print("No data found.")
        return

    # Ensure xGs
    print("Ensuring xG predictions...")
    df_data, _, _ = analyze._predict_xgs(df_data)

    # Identify all teams
    # We can find teams by looking at home_abb / away_abb
    teams = sorted(pd.concat([df_data['home_abb'], df_data['away_abb']]).unique())
    # For testing, limit to a few teams
    # teams = ['PHI']
    print(f"Found {len(teams)} teams: {teams}")

    # Condition for analysis (5v5)
    condition = {'game_state': ['5v5'], 'is_net_empty': [0]}
    
    # --- PASS 1: Calculate Stats (Game-Centric Optimization) ---
    master_csv_path = os.path.join(league_out_dir, 'league_player_stats.csv')
    
    if os.path.exists(master_csv_path):
        print(f"\n--- PASS 1: Skipped (Found existing stats at {master_csv_path}) ---")
        df_master = pd.read_csv(master_csv_path)
        print(f"Loaded {len(df_master)} players from CSV.")
        all_player_stats = df_master.to_dict('records')
    else:
        print("\n--- PASS 1: Calculating Player Stats (Game-Centric) ---")
        all_player_stats = []
        
        # Group data by game_id to iterate efficiently
        if 'game_id' not in df_data.columns:
            print("Error: 'game_id' not found in data.")
            return

        # Get unique games
        game_ids = sorted(df_data['game_id'].unique())
        print(f"Found {len(game_ids)} games to process.")

        # Helper for intersection (copied from timing._intersect_two to avoid internal import issues if any)
        def _intersect_intervals(a, b):
            res = []
            i = j = 0
            # Ensure sorted
            a = sorted(a)
            b = sorted(b)
            while i < len(a) and j < len(b):
                s1, e1 = a[i]
                s2, e2 = b[j]
                s = max(s1, s2)
                e = min(e1, e2)
                if e > s:
                    res.append((s, e))
                if e1 < e2:
                    i += 1
                else:
                    j += 1
            return res

        for i, game_id in enumerate(game_ids):
            if (i+1) % 10 == 0:
                print(f"Processing game {i+1}/{len(game_ids)}: {game_id}...")
                
            # 1. Load Shifts for Game (Once)
            # Use timing._get_shifts_df which now has in-memory caching
            df_shifts = timing._get_shifts_df(int(game_id))
            if df_shifts.empty:
                continue
                
            # 2. Compute Common 5v5 Intervals (Once)
            # This uses the exact same logic as analyze.players -> timing.compute_game_timing
            # We pass the condition to get the 5v5 intervals for this game.
            # Note: compute_intervals_for_game handles "Away" logic if 'team' is in condition.
            # Here we want global 5v5, so we don't pass 'team'.
            game_intervals_res = timing.compute_intervals_for_game(game_id, condition)
            
            # Extract the 5v5 intervals
            # The structure is {'intervals_per_condition': {'game_state': [...]}}
            # If we had multiple conditions they would be intersected.
            # Here we just have game_state='5v5' and is_net_empty=0
            # compute_intervals_for_game intersects them automatically if they are in the condition dict.
            # However, compute_intervals_for_game returns 'intersection_intervals' which is the intersection of ALL conditions passed.
            common_intervals = game_intervals_res.get('intersection_intervals', [])
            
            if not common_intervals:
                continue
                
            # 3. Identify Players in Game
            # We can get players from the shift chart
            players_in_game = df_shifts['player_id'].unique()
            
            # We also need to know which team each player is on for this game to calculate xG For/Against correctly
            # We can get this from the shift chart too
            # Create a map: pid -> team_id
            # We can use the first row for each player
            p_team_map = df_shifts.groupby('player_id')['team_id'].first().to_dict()
            
            # 4. Process Each Player
            for pid in players_in_game:
                try:
                    # Intersect player's shifts with common 5v5 intervals
                    # Get player's shifts
                    p_shifts_df = df_shifts[df_shifts['player_id'] == pid]
                    if p_shifts_df.empty:
                        continue
                    
                    # Convert player shifts to list of (start, end)
                    p_intervals = list(zip(p_shifts_df['start_total_seconds'], p_shifts_df['end_total_seconds']))
                    
                    # Intersect
                    final_intervals = _intersect_intervals(common_intervals, p_intervals)
                    
                    if not final_intervals:
                        continue
                    
                    # Calculate TOI
                    toi = sum(e - s for s, e in final_intervals)
                    if toi <= 0:
                        continue
                    
                    # Construct intervals_input for xgs_map
                    # xgs_map expects: {'per_game': {game_id: {'intersection_intervals': [...]}}}
                    # (via _apply_intervals logic)
                    intervals_input = {
                        'per_game': {
                            game_id: {
                                'intersection_intervals': final_intervals,
                                # We must ensure _apply_intervals uses these.
                                # It looks for 'sides' -> 'team' -> ... or 'intersection_intervals'
                                # So this structure is correct.
                            }
                        }
                    }
                    
                    # Determine player's team for xG splitting
                    p_team_id = p_team_map.get(pid)
                    
                    # Call xgs_map
                    # IMPORTANT: We pass 'team': p_team_id in condition so xgs_map knows which is "For" and "Against".
                    # BUT we do NOT pass 'player_id' in condition, because we want ON-ICE stats (all events),
                    # and we rely on intervals_input to filter by time (when player was on ice).
                    p_cond = condition.copy()
                    p_cond['team'] = p_team_id
                    
                    # We need to pass a dataframe containing this game's events
                    # We can filter the main df_data to this game
                    df_game = df_data[df_data['game_id'] == game_id]
                    
                    _, _, _, p_stats = analyze.xgs_map(
                        season=season,
                        data_df=df_game,
                        condition=p_cond,
                        out_path=None,
                        return_heatmaps=False,
                        show=False,
                        total_seconds=toi,
                        use_intervals=True,
                        intervals_input=intervals_input,
                        stats_only=True # Optimization
                    )
                    
                    if p_stats:
                        # Add metadata
                        p_stats['player_id'] = pid
                        p_stats['game_id'] = game_id
                        
                        # Resolve Team Abbreviation
                        # We can look it up from df_game
                        # If p_team_id matches home_id, use home_abb, else away_abb
                        # Just pick first row
                        if not df_game.empty:
                            r = df_game.iloc[0]
                            if str(r.get('home_id')) == str(p_team_id):
                                p_stats['team'] = r.get('home_abb')
                            elif str(r.get('away_id')) == str(p_team_id):
                                p_stats['team'] = r.get('away_abb')
                            else:
                                p_stats['team'] = 'UNK'
                        else:
                             p_stats['team'] = 'UNK'
                             
                        # Calculate per-60s for this game (optional, but good for debugging)
                        p_stats['xg_for'] = p_stats.get('team_xgs', 0.0)
                        p_stats['xg_against'] = p_stats.get('other_xgs', 0.0)
                        p_stats['toi_sec'] = toi
                        
                        all_player_stats.append(p_stats)
                        
                except Exception as e:
                    # print(f"Error processing player {pid} in game {game_id}: {e}")
                    continue

        if not all_player_stats:
            print("No player stats calculated.")
            return

        # Aggregate stats per player
        print("Aggregating stats per player...")
        df_raw = pd.DataFrame(all_player_stats)
        
        # Group by player_id
        # Sum: xg_for, xg_against, toi_sec, games_played (count)
        # First, ensure we have a name. We can get name from df_data or just use what we have?
        # xgs_map doesn't return player name.
        # We need to fetch player names.
        # We can build a map from df_data
        p_name_map = df_data[['player_id', 'player_name']].dropna().drop_duplicates('player_id').set_index('player_id')['player_name'].to_dict()
        
        agg_stats = []
        for pid, grp in df_raw.groupby('player_id'):
            tot_xg_for = grp['xg_for'].sum()
            tot_xg_against = grp['xg_against'].sum()
            tot_toi = grp['toi_sec'].sum()
            games = grp['game_id'].nunique()
            team = grp['team'].mode().iloc[0] if not grp['team'].empty else 'UNK'
            name = p_name_map.get(pid, f"Player {pid}")
            
            if tot_toi > 0:
                xg_for_60 = (tot_xg_for / tot_toi) * 3600
                xg_against_60 = (tot_xg_against / tot_toi) * 3600
            else:
                xg_for_60 = 0
                xg_against_60 = 0
                
            agg_stats.append({
                'player_id': pid,
                'name': name,
                'team': team,
                'xg_for': tot_xg_for,
                'xg_against': tot_xg_against,
                'toi_sec': tot_toi,
                'xg_for_60': xg_for_60,
                'xg_against_60': xg_against_60,
                'games_played': games
            })
        
        all_player_stats = agg_stats # Replace with aggregated list for CSV generation

        # Create Master DataFrame
        df_master = pd.DataFrame(all_player_stats)
        df_master.to_csv(master_csv_path, index=False)
        print(f"Saved master stats to {master_csv_path}")

    # --- Calculate Percentiles ---
    print("\n--- Calculating Percentiles ---")
    # We calculate percentiles for xG For, xG Against, Rel Off, Rel Def
    # We should filter for min_games before calculating percentiles? 
    # Usually percentiles are based on "qualified" players.
    # Let's use min_games=5 for percentile basis to avoid noise from 1-game wonders.
    min_games_for_percentile = 5
    qual_df = df_master[df_master['games_played'] >= min_games_for_percentile]
    
    percentiles = {} # pid -> {off: val, def: val}
    
    for pid in df_master['player_id'].unique():
        # Get player's stats
        p_row = df_master[df_master['player_id'] == pid]
        if p_row.empty: continue
        p_row = p_row.iloc[0]
        
        # Calculate percentile rank within qualified players
        # If player is not qualified, we still calculate their rank against qualifieds?
        # Or just assign N/A? Let's calculate against qualifieds.
        
        # Offense (Higher is better)
        off_val = p_row['xg_for_60']
        off_pct = (qual_df['xg_for_60'] < off_val).mean() * 100
        
        # Defense (Lower is better) - so we count how many are GREATER than this value (worse defense)
        # Wait, percentile usually means "better than X%".
        # For defense, lower xGA is better.
        # So if I have 2.0 xGA, and 90% of league has > 2.0, I am in 90th percentile?
        # Yes. (qual_df['xg_against_60'] > def_val).mean() * 100
        def_val = p_row['xg_against_60']
        def_pct = (qual_df['xg_against_60'] > def_val).mean() * 100
        
        percentiles[pid] = {
            'off': off_pct, # Pass as float
            'def': def_pct  # Pass as float
        }

    # --- PASS 2: Generate Maps ---
    print("\n--- PASS 2: Generating Maps ---")
    
    # Filter for qualified players
    qual_pids = df_master[df_master['games_played'] >= min_games_for_percentile]['player_id'].unique().tolist()
    print(f"Generating maps for {len(qual_pids)} qualified players (>= {min_games_for_percentile} games).")
    
    for team in teams:
        print(f"Processing {team} (Plotting)...")
        
        # Filter qualified players for THIS team
        team_pids = df_master[
            (df_master['games_played'] >= min_games_for_percentile) & 
            (df_master['team'] == team)
        ]['player_id'].unique().tolist()
        
        if not team_pids:
            print(f"No qualified players found for {team}.")
            continue
            
        analyze.players(
            season=season,
            team=team,
            player_ids=team_pids, # Only process qualified players for THIS team
            condition=condition,
            out_dir=os.path.join(out_dir_base, f'{season}/{team}'),
            data_df=df_data,
            plot=True,
            percentiles=percentiles,
            min_games=min_games_for_percentile
        )

    # --- League-Wide Scatter Plot ---
    print("\n--- Generating League-Wide Scatter Plot ---")
    generate_league_scatter(df_master, league_out_dir, season, min_games=5)

def generate_league_scatter(df, out_dir, season, min_games=5):
    # Filter
    df_plot = df[df['games_played'] >= min_games].copy()
    print(f"League Scatter: {len(df_plot)} players (min_games={min_games})")
    
    if df_plot.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot all points
    # Color by position? We don't have position in df_master yet (it's in p_stats if we added it, but we didn't explicitly)
    # We can color by Team? Too many colors.
    # Just uniform color, maybe density?
    ax.scatter(df_plot['xg_for_60'], df_plot['xg_against_60'], alpha=0.5, c='gray', s=30, label='Players')
    
    # Smart Labeling
    # Label top 5% offense, top 5% defense (low xGA), top 5% bad defense, top 5% bad offense?
    # Or just outliers from the diagonal?
    # Let's label top 3 players in each "quadrant" relative to median.
    
    med_off = df_plot['xg_for_60'].median()
    med_def = df_plot['xg_against_60'].median()
    
    # Add median lines
    ax.axvline(med_off, color='k', linestyle=':', alpha=0.3)
    ax.axhline(med_def, color='k', linestyle=':', alpha=0.3)
    
    # Identify outliers to label
    # We can calculate a "distance from center" or "impact score"
    # Impact = (Off - Med_Off) - (Def - Med_Def) ? (since lower Def is better)
    # Let's label top 10 by "Net xG Diff" (xGF - xGA)
    df_plot['net_xg'] = df_plot['xg_for_60'] - df_plot['xg_against_60']
    top_net = df_plot.nlargest(10, 'net_xg')
    bottom_net = df_plot.nsmallest(5, 'net_xg')
    
    # Also label extreme offense/defense
    top_off = df_plot.nlargest(5, 'xg_for_60')
    top_def = df_plot.nsmallest(5, 'xg_against_60') # Low xGA
    
    labels_to_plot = pd.concat([top_net, bottom_net, top_off, top_def]).drop_duplicates('player_id')
    
    # Highlight labeled points
    ax.scatter(labels_to_plot['xg_for_60'], labels_to_plot['xg_against_60'], color='red', s=40, zorder=5)
    
    from adjustText import adjust_text
    texts = []
    for _, row in labels_to_plot.iterrows():
        t = ax.text(row['xg_for_60'], row['xg_against_60'], f"{row['name']} ({row['team']})", fontsize=8)
        texts.append(t)
        
    # Try to adjust text if library exists, otherwise just basic
    try:
        from adjustText import adjust_text
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))
    except ImportError:
        pass

    # Limits and Unity Line
    vals = pd.concat([df_plot['xg_for_60'], df_plot['xg_against_60']])
    max_val = vals.max()
    min_val = vals.min()
    padding = (max_val - min_val) * 0.05
    limit_max = max_val + padding
    limit_min = max(0, min_val - padding)
    
    ax.set_xlim(limit_min, limit_max)
    ax.set_ylim(limit_min, limit_max)
    ax.invert_yaxis()
    ax.plot([limit_min, limit_max], [limit_min, limit_max], color='gray', linestyle='--', alpha=0.5, label='xGF = xGA')
    
    ax.set_aspect('equal')
    ax.set_xlabel('xG For / 60')
    ax.set_ylabel('xG Against / 60')
    ax.set_title(f'League-Wide Player xG Rates (5v5) - {season}\n(Min {min_games} Games)')
    ax.grid(True, alpha=0.3)
    
    out_path = os.path.join(out_dir, 'league_scatter.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved league scatter to {out_path}")

if __name__ == "__main__":
    run_analysis()
