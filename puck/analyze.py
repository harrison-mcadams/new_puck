## Various different analyses


from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import pickle

# Import our custom viewer
try:
    from .view_utils import view_df
except ImportError:
    def view_df(df, title=""): print(f"view_df not available: {title}")

# Import Config for valid Data Directory
try:
    from . import config as puck_config
except ImportError:
    try:
        import config as puck_config
    except ImportError:
        # Provide a dummy config if strictly standalone and config missing (rare)
        class DummyConfig:
            DATA_DIR = 'data'
            ANALYSIS_DIR = 'analysis'
        puck_config = DummyConfig()

def _resolve_baseline_path(season: str, condition: Optional[dict]) -> str:
    """
    Resolve the path to the league baseline directory based on the condition.
    
    Args:
        season: Season string.
        condition: Filtering condition.
        
    Returns:
        str: Path to the directory containing the baseline.
    """
    import os
    
    # Default to 5v5 if no condition
    # Default to 5v5 if no condition
    if condition is None:
        return os.path.join(puck_config.ANALYSIS_DIR, 'league', season, '5v5')
        
    # Check for standard game states
    game_state = condition.get('game_state')
    is_net_empty = condition.get('is_net_empty')
    
    cond_name = '5v5' # Default
    
    if game_state:
        # Handle list or single value
        gs = game_state[0] if isinstance(game_state, list) and game_state else game_state
        if isinstance(gs, str):
            cond_name = gs
            
    # Check for empty net
    if is_net_empty == [1] or is_net_empty == 1:
        # If empty net is explicitly requested, we might have a different folder structure
        # But currently run_league_stats.py uses '5v5', '5v4', etc. which imply no empty net usually
        # unless specified.
        # For now, let's stick to the game_state name if it's 5v5, 5v4, 4v5
        pass
        
    # Construct path: analysis/league/{season}/{cond_name}
    # This matches run_league_stats.py structure
    return os.path.join(puck_config.ANALYSIS_DIR, 'league', season, cond_name)

def locate_season_csv(season: str, csv_path: str = None) -> str:
    """Find a CSV for the given season using a prioritized list of
    candidates, then a recursive search of data/ if necessary.
    """
    from pathlib import Path
    if csv_path:
        p = Path(csv_path)
        if p.exists():
            return str(p)
    # Prioritize data/ directory and support both naming conventions
    candidates = [
        # Primary flat structure
        Path('data') / season / f'{season}_df.csv',
        Path('data') / season / f'{season}.csv',
        # Legacy processed path (deprecated but kept for fallback just in case)
        Path('data') / 'processed' / season / f'{season}.csv',
        Path('data') / 'processed' / season / f'{season}_df.csv',
    ]
    for c in candidates:
        try:
            if c.exists():
                return str(c)
        except Exception:
            continue
    # fallback: find any CSV under data/ matching season
    data_dir = Path('data')
    if data_dir.exists():
        found = list(data_dir.rglob(f'*{season}*.csv'))
        if found:
            return str(found[0])
    raise FileNotFoundError(f'Could not locate a CSV for season {season}.')

def players(season: str = '20252026',
            team: Optional[str] = None,
            player_ids: Optional[list] = None,
            game_id: Optional[str] = None,
            time_scope: str = 'season',
            condition: Optional[dict] = None,
            out_dir: Optional[str] = None,
            min_games: int = 5,
            data_df: Optional[pd.DataFrame] = None,
            plot: bool = True,
            percentiles: Optional[dict] = None):
    """
    Analyze player performance (xG For/Against) for a list of players or a whole team.

    Args:
        season: Season string (e.g., '20252026').
        team: Team abbreviation to analyze all players for (e.g., 'PHI').
        player_ids: List of specific player IDs to analyze.
        game_id: Specific game ID if time_scope is 'game'.
        time_scope: 'season' or 'game' (or 'career' - treated as all loaded data).
        condition: Filtering condition (e.g., {'game_state': ['5v5']}).
        out_dir: Output directory.
        min_games: Minimum games played to include in scatter plot (default 5).
        data_df: Optional pre-loaded dataframe to avoid reloading.
        plot: Whether to generate plots (default True).
        percentiles: Optional dictionary of percentiles {player_id: {'off': val, 'def': val}}.
    """
    import os
    import json
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from .plot import add_summary_text, rink_goal_xs
    from .rink import draw_rink
    from . import timing

    # 1. Setup Output Directory
    if out_dir is None:
        if team:
            out_dir = os.path.join('static', f'{season}_{team}_players_analysis')
        else:
            out_dir = os.path.join('static', f'{season}_players_analysis')
    os.makedirs(out_dir, exist_ok=True)

    print(f"players: Starting analysis. Scope={time_scope}, Team={team}, Condition={condition}", flush=True)
    print(f"DEBUG: analyze.players called", flush=True)

    # 2. Load Data
    # We load the full dataset once to avoid reloading per player
    df_data = None
    if time_scope == 'game' and game_id:
        # Load specific game
        print(f"players: Fetching data for game {game_id}...")
        _, _, df_data, _ = xgs_map(season=season, game_id=game_id, return_filtered_df=True, show=False, return_heatmaps=False)
    elif data_df is not None:
        # Use provided dataframe
        df_data = data_df
    else:
        # Load season data
        print(f"players: Loading data for season {season}...")
        df_data = timing.load_season_df(season)
    
    if df_data is None or df_data.empty:
        print("players: No data found.")
        return

    # Ensure xGs are present (predict if needed)
    print("players: Ensuring xG predictions...")
    df_data, _, _ = _predict_xgs(df_data)

    # 3. Resolve Players
    target_pids = set()
    if player_ids:
        target_pids.update(player_ids)
    
    if team:
        # Helper to identify team ID from abbreviation
        team_id = None
        # Try to find a row with this team
        sample = df_data[df_data['home_abb'] == team]
        if not sample.empty:
            team_id = sample.iloc[0]['home_id']
        else:
            sample = df_data[df_data['away_abb'] == team]
            if not sample.empty:
                team_id = sample.iloc[0]['away_id']
        
        if team_id:
            # Get all player_ids where team_id matches
            # We filter for events where the team matches the requested team
            team_events = df_data[df_data['team_id'] == team_id]
            found_pids = team_events['player_id'].dropna().unique().tolist()
            target_pids.update(found_pids)
            print(f"players: Found {len(found_pids)} players for team {team}")
        else:
            print(f"players: Could not resolve team ID for {team}")

    if not target_pids:
        print("players: No players identified to analyze.")
        return

    # 4. Load League Baseline (for relative maps)
    print("players: Loading league baseline...")
    
    # Resolve baseline path based on condition
    baseline_path = _resolve_baseline_path(season, condition)
    print(f"players: Using baseline path: {baseline_path}")
    
    baseline_res = league(season=season, mode='load', condition=condition, baseline_path=baseline_path)
    league_map = baseline_res.get('combined_norm')
    league_map_right = baseline_res.get('combined_norm_right')
    
    # Scale Baseline to Per 60
    if league_map is not None:
        league_map = league_map * 3600.0
    if league_map_right is not None:
        league_map_right = league_map_right * 3600.0
        
    league_xg_per60 = baseline_res.get('stats', {}).get('xg_per60', 0.0)

    # 5. Analyze Each Player
    player_stats = []
    
    # Base condition
    base_cond = condition.copy() if condition else {}
    base_cond.pop('player_id', None)
    base_cond.pop('team', None) 
    
    # Build a simple ID->Name map from the dataframe
    pid_name_map = {}
    pid_num_map = {}
    if 'player_name' in df_data.columns:
        cols = ['player_id', 'player_name']
        # Check for potential number columns
        num_col = None
        for c in ['player_number', 'sweater_number', 'number', 'jersey_number']:
            if c in df_data.columns:
                num_col = c
                cols.append(c)
                break
        
        pairs = df_data[cols].dropna(subset=['player_id']).drop_duplicates('player_id')
        pid_name_map = dict(zip(pairs['player_id'], pairs['player_name']))
        if num_col:
            pid_num_map = dict(zip(pairs['player_id'], pairs[num_col]))

    print(f"players: Analyzing {len(target_pids)} players...")
    
    # Pre-fetch player details if needed
    from .nhl_api import get_player_details
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import os

    # Helper function for parallel execution
    def _process_single_player(pid, pname, pnum, df_data, base_cond, team, out_dir, season, league_map, league_map_right, plot, min_games):
        try:
            # Optimization: Filter data to only games where player played
            p_games = df_data[df_data['player_id'] == pid]['game_id'].unique()
            
            if len(p_games) < min_games:
                return None
                
            p_df = df_data[df_data['game_id'].isin(p_games)].copy()
            
            # Infer team if not provided
            p_team = team
            if not p_team:
                p_events = p_df[p_df['player_id'] == pid]
                if not p_events.empty:
                    top_team_id = p_events['team_id'].mode()
                    if not top_team_id.empty:
                        p_team = top_team_id[0]
                        try:
                            row = p_df[(p_df['home_id'] == p_team) | (p_df['away_id'] == p_team)].iloc[0]
                            if row['home_id'] == p_team:
                                p_team = row['home_abb']
                            else:
                                p_team = row['away_abb']
                        except Exception:
                            pass

            # Setup condition for this player
            p_cond = base_cond.copy()
            p_cond['player_id'] = pid
            if p_team:
                p_cond['team'] = p_team
            
            # Output path
            p_out_path = os.path.join(out_dir, f"{pid}_map.png")
            
            # Calculate robust timing
            timing_res = timing.compute_game_timing(p_df, p_cond, season=season)
            t_seconds = timing_res.get('aggregate', {}).get('intersection_seconds_total', 0.0)
            
            _, ret_heat, _, p_stats = xgs_map(
                season=season,
                data_df=p_df,
                condition=p_cond,
                out_path=None,
                return_heatmaps=True,
                show=False,
                total_seconds=t_seconds,
                use_intervals=True,
                title=pname,
                stats_only=not plot
            )
            
            if not p_stats:
                return None

            xg_for = p_stats.get('team_xgs', 0.0)
            xg_ag = p_stats.get('other_xgs', 0.0)
            seconds = p_stats.get('team_seconds', 0.0)
            
            if seconds <= 0:
                return None

            xg_for_60 = (xg_for / seconds) * 3600
            xg_ag_60 = (xg_ag / seconds) * 3600
            
            # Relative Map Calculation
            rel_path = None
            rel_off_60 = None
            rel_def_60 = None
            
            team_map = None
            if isinstance(ret_heat, dict):
                team_map = ret_heat.get('team') or ret_heat.get('home')
            
            if team_map is not None and league_map is not None:
                other_map = ret_heat.get('other') or ret_heat.get('not_team')
                
                combined_rel_map, rel_off_pct, rel_def_pct, rel_off_60, rel_def_60 = compute_relative_map(
                    team_map, league_map, seconds, other_map, seconds,
                    league_baseline_right=league_map_right
                )
                
                if plot:
                    # Plot Relative Map
                    rel_path = os.path.join(out_dir, f"{pid}_relative.png")
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    draw_rink(ax=ax)
                    
                    from matplotlib.colors import SymLogNorm
                    norm = SymLogNorm(linthresh=1e-5, linscale=1.0, vmin=-0.0006, vmax=0.0006, base=10)
                    cbar_ticks = [-0.0006, -0.0001, -0.00001, 0, 0.00001, 0.0001, 0.0006]
                    cbar_ticklabels = ['High -', 'Med -', 'Low -', 'Avg', 'Low +', 'Med +', 'High +']
                    
                    gx = np.arange(-100.0, 100.0 + 1.0, 1.0)
                    gy = np.arange(-42.5, 42.5 + 1.0, 1.0)
                    extent = (gx[0] - 0.5, gx[-1] + 0.5, gy[0] - 0.5, gy[-1] + 0.5)
                    
                    cmap = plt.get_cmap('RdBu_r')
                    try: cmap.set_bad(color=(1,1,1,0)) 
                    except: pass
                    
                    m = np.ma.masked_invalid(combined_rel_map)
                    im = ax.imshow(m, extent=extent, origin='lower', cmap=cmap, norm=norm)
                    
                    txt_stats = p_stats.copy()
                    txt_stats['home_xg'] = xg_for
                    txt_stats['away_xg'] = xg_ag
                    txt_stats['have_xg'] = True
                    txt_stats['home_goals'] = p_stats.get('team_goals', 0)
                    txt_stats['away_goals'] = p_stats.get('other_goals', 0)
                    txt_stats['home_attempts'] = p_stats.get('team_attempts', 0)
                    txt_stats['away_attempts'] = p_stats.get('other_attempts', 0)
                    
                    # Add percentiles (passed in p_stats if available, or we need to handle them)
                    # Note: percentiles are not passed to this helper, we need to add them to p_stats before calling or inside
                    # Actually, percentiles are in the 'percentiles' dict in the main function.
                    # We can pass them in.
                    # BUT wait, we can't pass the whole percentiles dict to every worker efficiently?
                    # It's small enough.
                    
                    h_att = txt_stats['home_attempts']
                    a_att = txt_stats['away_attempts']
                    tot_att = h_att + a_att
                    txt_stats['home_shot_pct'] = 100.0 * h_att / tot_att if tot_att > 0 else 0.0
                    txt_stats['away_shot_pct'] = 100.0 * a_att / tot_att if tot_att > 0 else 0.0
                    
                    cond_str = "All"
                    if base_cond: # Use base_cond as condition
                         # ... (reconstruct cond_str logic)
                         pass

                    # We need to reconstruct cond_str logic or pass it in.
                    # Let's just pass it in or simplify.
                    
                    add_summary_text(
                        ax=ax,
                        stats=txt_stats,
                        main_title=pname,
                        is_season_summary=True,
                        team_name=team or "UNK",
                        full_team_name=pname,
                        filter_str=str(base_cond) # Simplified for now
                    )
                    ax.axis('off')
                    
                    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
                    cbar.set_label('Relative xG/60 Difference', rotation=270, labelpad=20)
                    if cbar_ticks:
                        cbar.set_ticks(cbar_ticks)
                        cbar.set_ticklabels(cbar_ticklabels)
                    
                    fig.savefig(rel_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)

                    # Save Raw Map
                    try:
                        from .plot import plot_events
                        ret_pe = plot_events(
                            p_df,
                            out_path=p_out_path,
                            title=pname,
                            summary_stats=txt_stats,
                            plot_kwargs={
                                'is_season_summary': True,
                                'filter_str': str(base_cond),
                                'team_for_heatmap': p_team,
                            }
                        )
                        if ret_pe and len(ret_pe) >= 1:
                            plt.close(ret_pe[0])
                    except Exception as e:
                        print(f"players: Failed to save map for {pid}: {e}")

            return {
                'player_id': pid,
                'name': pname,
                'team': team,
                'xg_for': xg_for,
                'xg_against': xg_ag,
                'toi_sec': seconds,
                'xg_for_60': xg_for_60,
                'xg_against_60': xg_ag_60,
                'rel_off_60': rel_off_60,
                'rel_def_60': rel_def_60,
                'map_path': rel_path,
                'games_played': len(p_games)
            }
        except Exception as e:
            print(f"Error processing {pid}: {e}")
            return None

    # Prepare tasks
    tasks = []
    for pid in target_pids:
        pid = int(pid)
        pname = pid_name_map.get(pid)
        pnum = pid_num_map.get(pid)
        
        # ... (name resolution logic same as before) ...
        if not pname or not pnum:
             # ... (logic to fetch name) ...
             # For parallel, maybe skip fetching name if missing to avoid API calls in loop?
             # Or fetch before?
             pass
        
        if not pname: pname = f"Player {pid}"
        if pnum: pname = f"{pname} #{int(pnum)}"
        
        tasks.append((pid, pname, pnum))

    # Run in parallel
    # NOTE: We can't pickle the whole df_data efficiently if it's huge.
    # But on Mac (spawn), it might copy. On fork (default for some), it's COW.
    # Python 3.8+ on Mac defaults to 'spawn'.
    # This might be slow to copy df_data.
    # Alternative: Use threads? Matplotlib is not thread safe.
    # We stick to sequential for now but optimize the loop?
    
    # Actually, the user asked to speed it up.
    # The bottleneck is likely I/O and plotting.
    # Let's try to just optimize the loop logic first without full multiprocessing refactor
    # because passing df_data to workers is heavy.
    
    for pid in target_pids:
    # ... (original loop) ...
        pid = int(pid)
        pname = pid_name_map.get(pid)
        pnum = pid_num_map.get(pid)
        
        # If name or number missing, try to fetch from API using a game_id
        if not pname or not pnum:
            # Find a game this player played in
            p_games = df_data[df_data['player_id'] == pid]['game_id'].unique()
            # Try games in reverse order until we find details
            for gid in reversed(p_games):
                try:
                    details = get_player_details(gid)
                    if pid in details:
                        if not pname: pname = details[pid].get('name')
                        if not pnum: pnum = details[pid].get('number')
                        # If we found both, break
                        if pname and pnum:
                            break
                except Exception:
                    continue
        
        if not pname: pname = f"Player {pid}"
        if pnum:
            pname = f"{pname} #{int(pnum)}"
        
        # Optimization: Filter data to only games where player played
        # This prevents computing timing for the entire season for every player
        p_games = df_data[df_data['player_id'] == pid]['game_id'].unique()
        
        if len(p_games) < min_games:
            # print(f"players: Skipping {pname} ({len(p_games)} games < {min_games})")
            continue
            
        p_df = df_data[df_data['game_id'].isin(p_games)].copy()
        
        # Infer team if not provided
        p_team = team
        if not p_team:
            # Find the team_id associated with the player in p_df
            # We look for events where player_id == pid and get the team_id
            p_events = p_df[p_df['player_id'] == pid]
            if not p_events.empty:
                # Get most common team_id
                top_team_id = p_events['team_id'].mode()
                if not top_team_id.empty:
                    p_team = top_team_id[0]
                    # Also try to get abbreviation if possible
                    # (This is optional but helpful for filenames)
                    # We can look up home_abb/away_abb where home_id/away_id matches p_team
                    try:
                        row = p_df[(p_df['home_id'] == p_team) | (p_df['away_id'] == p_team)].iloc[0]
                        if row['home_id'] == p_team:
                            p_team = row['home_abb']
                        else:
                            p_team = row['away_abb']
                    except Exception:
                        pass

        # Setup condition for this player
        p_cond = base_cond.copy()
        p_cond['player_id'] = pid
        if p_team:
            p_cond['team'] = p_team
        
        try:
            # Output path for this player's map
            # Use p_team for filename if available
            # Output path for this player's map
            # Since we are likely in a team folder, we can simplify the filename
            # But to be safe and consistent, we'll keep the team prefix or just use PID
            # User requested "player_ids", let's use just PID if we are in a team folder context?
            # Actually, let's stick to {pid}_map.png to be clean inside the team folder.
            p_out_path = os.path.join(out_dir, f"{pid}_map.png")
            
            # Calculate robust timing for player
            timing_res = timing.compute_game_timing(p_df, p_cond, season=season)
            t_seconds = timing_res.get('aggregate', {}).get('intersection_seconds_total', 0.0)
            
            if t_seconds is None:
                print(f"DEBUG: t_seconds is None for pid={pid}", flush=True)
                print(f"DEBUG: timing_res keys: {list(timing_res.keys())}", flush=True)
                print(f"DEBUG: aggregate: {timing_res.get('aggregate')}", flush=True)
                t_seconds = 0.0
            try:
                t_seconds = float(t_seconds)
            except Exception:
                t_seconds = 0.0

            _, ret_heat, _, p_stats = xgs_map(
                season=season,
                data_df=p_df,
                condition=p_cond,
                out_path=None, # Don't save yet, we'll save manually with correct summary text
                return_heatmaps=True,
                show=False,
                total_seconds=t_seconds,
                use_intervals=True, # Force interval filtering to get On-Ice data
                title=pname,
                stats_only=not plot # Optimization: skip heatmap/plotting if we aren't plotting
            )
            
            if not p_stats:
                continue

            xg_for = p_stats.get('team_xgs', 0.0)
            xg_ag = p_stats.get('other_xgs', 0.0)
            seconds = p_stats.get('team_seconds', 0.0)
            
            if seconds <= 0:
                continue

            xg_for_60 = (xg_for / seconds) * 3600
            xg_ag_60 = (xg_ag / seconds) * 3600
            
            # Relative Map Calculation
            team_map = None
            if isinstance(ret_heat, dict):
                team_map = ret_heat.get('team')
                if team_map is None:
                    team_map = ret_heat.get('home')
            
            rel_path = None
            if team_map is not None and league_map is not None:
                other_map = ret_heat.get('other')
                if other_map is None:
                    other_map = ret_heat.get('not_team')
                
                # DEBUG: Inspect maps
                # tm_sum = np.nansum(team_map) if team_map is not None else 0
                # om_sum = np.nansum(other_map) if other_map is not None else 0
                # lm_sum = np.nansum(league_map) if league_map is not None else 0
                # print(f"DEBUG: Relative Map Inputs for {pid}:")
                # print(f"  Seconds: {seconds}")
                # print(f"  Team Map Sum: {tm_sum:.4f}, Max: {np.nanmax(team_map) if team_map is not None else 0:.4f}")
                # print(f"  Other Map Sum: {om_sum:.4f}, Max: {np.nanmax(other_map) if other_map is not None else 0:.4f}")
                # print(f"  League Map Sum: {lm_sum:.4f}, Max: {np.nanmax(league_map) if league_map is not None else 0:.4f}")

                combined_rel_map, rel_off_pct, rel_def_pct, rel_off_60, rel_def_60 = compute_relative_map(
                    team_map, league_map, seconds, other_map, seconds,
                    league_baseline_right=league_map_right
                )
                
                if plot:
                    # Plot Relative Map
                    rel_path = os.path.join(out_dir, f"{pid}_relative.png")
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    draw_rink(ax=ax)
                    
                    # Use SymLogNorm for relative map (consistent with season analysis)
                    from matplotlib.colors import SymLogNorm
                    norm = SymLogNorm(linthresh=1e-5, linscale=1.0, vmin=-0.0006, vmax=0.0006, base=10)
                    
                    cbar_ticks = [-0.0006, -0.0001, -0.00001, 0, 0.00001, 0.0001, 0.0006]
                    cbar_ticklabels = ['High -', 'Med -', 'Low -', 'Avg', 'Low +', 'Med +', 'High +']
                    
                    gx = np.arange(-100.0, 100.0 + 1.0, 1.0)
                    gy = np.arange(-42.5, 42.5 + 1.0, 1.0)
                    extent = (gx[0] - 0.5, gx[-1] + 0.5, gy[0] - 0.5, gy[-1] + 0.5)
                    
                    cmap = plt.get_cmap('RdBu_r')
                    try: cmap.set_bad(color=(1,1,1,0)) 
                    except: pass
                    
                    m = np.ma.masked_invalid(combined_rel_map)
                    im = ax.imshow(m, extent=extent, origin='lower', cmap=cmap, norm=norm)
                    
                    txt_stats = p_stats.copy()
                    txt_stats['home_xg'] = xg_for
                    txt_stats['away_xg'] = xg_ag
                    txt_stats['have_xg'] = True
                    
                    txt_stats['home_goals'] = p_stats.get('team_goals', 0)
                    txt_stats['away_goals'] = p_stats.get('other_goals', 0)
                    txt_stats['home_attempts'] = p_stats.get('team_attempts', 0)
                    txt_stats['away_attempts'] = p_stats.get('other_attempts', 0)
                    
                    # Add relative stats for summary text
                    txt_stats['rel_off_pct'] = rel_off_pct
                    txt_stats['rel_def_pct'] = rel_def_pct
                    
                    # Add percentiles if available
                    if percentiles and pid in percentiles:
                        txt_stats['off_percentile'] = percentiles[pid].get('off')
                        txt_stats['def_percentile'] = percentiles[pid].get('def')
                    
                    # Calculate shot percentages if not present
                    h_att = txt_stats['home_attempts']
                    a_att = txt_stats['away_attempts']
                    tot_att = h_att + a_att
                    if tot_att > 0:
                        txt_stats['home_shot_pct'] = 100.0 * h_att / tot_att
                        txt_stats['away_shot_pct'] = 100.0 * a_att / tot_att
                    else:
                        txt_stats['home_shot_pct'] = 0.0
                        txt_stats['away_shot_pct'] = 0.0
                    
                    # Format condition string nicely
                    cond_str = "All"
                    if condition:
                        parts = []
                        if 'game_state' in condition:
                            parts.append(",".join(condition['game_state']))
                        if 'is_net_empty' in condition:
                            val = condition['is_net_empty']
                            if val == [0]: parts.append("No Empty Net")
                            elif val == [1]: parts.append("Empty Net")
                        cond_str = " | ".join(parts) if parts else str(condition)

                    add_summary_text(
                        ax=ax,
                        stats=txt_stats,
                        main_title=pname,
                        is_season_summary=True,
                        team_name=team or "UNK",
                        full_team_name=pname,
                        filter_str=cond_str
                    )
                    ax.axis('off')
                    
                    # Colorbar
                    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
                    cbar.set_label('Relative xG/60 Difference', rotation=270, labelpad=20)
                    if cbar_ticks:
                        cbar.set_ticks(cbar_ticks)
                        cbar.set_ticklabels(cbar_ticklabels)
                    
                    fig.savefig(rel_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    # print(f"players: Generated relative map for {pid} at {rel_path}")

                    # Save the raw map with consistent summary text
                    try:
                        from .plot import plot_events
                        
                        ret_pe = plot_events(
                            p_df,
                            out_path=p_out_path,
                            title=pname,
                            summary_stats=txt_stats, # Use txt_stats which has percentiles
                            plot_kwargs={
                                'is_season_summary': True,
                                'filter_str': cond_str,
                                'team_for_heatmap': p_team,
                            }
                        )
                        if ret_pe and len(ret_pe) >= 1:
                            plt.close(ret_pe[0])
                        # print(f"players: Generated map for {pid} at {p_out_path}")
                    except Exception as e:
                        print(f"players: Failed to save map for {pid}: {e}")

            player_stats.append({
                'player_id': pid,
                'name': pname,
                'team': team,
                'xg_for': xg_for,
                'xg_against': xg_ag,
                'toi_sec': seconds,
                'xg_for_60': xg_for_60,
                'xg_against_60': xg_ag_60,
                'rel_off_60': rel_off_60 if 'rel_off_60' in locals() else None,
                'rel_def_60': rel_def_60 if 'rel_def_60' in locals() else None,
                'map_path': rel_path,
                'games_played': len(p_games)
            })
            
        except Exception as e:
            print(f"players: Error analyzing player {pid}: {e}")
            import traceback
            traceback.print_exc()

    # 6. Aggregate and Plot Scatter
    if not player_stats:
        print("players: No player stats generated.")
        return []

    df_stats = pd.DataFrame(player_stats)
    
    # Save stats if plotting or if explicit output dir
    if plot or out_dir:
        csv_path = os.path.join(out_dir, 'player_stats.csv')
        df_stats.to_csv(csv_path, index=False)
        print(f"players: Saved stats to {csv_path}")

    if plot:
        try:
            # Match style of team scatter plot (generate_scatter_plot)
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Filter stats for scatter plot
            scatter_stats = df_stats[df_stats['games_played'] >= min_games]
            print(f"players: Scatter plot includes {len(scatter_stats)} players (min_games={min_games})")
            
            if scatter_stats.empty:
                print("players: No players meet min_games criteria for scatter plot.")
            else:
                vals = pd.concat([scatter_stats['xg_for_60'], scatter_stats['xg_against_60']])
                max_val = vals.max()
                min_val = vals.min()
                
                padding = (max_val - min_val) * 0.1 if (max_val - min_val) > 0 else 0.5
                limit_max = max_val + padding
                limit_min = max(0, min_val - padding)
                
                # Set limits explicitly to be equal
                ax.set_xlim(limit_min, limit_max)
                ax.set_ylim(limit_min, limit_max)
                
                # Invert Y axis (lower xGA is better)
                ax.invert_yaxis()
                
                # Unity line (x=y)
                ax.plot([limit_min, limit_max], [limit_min, limit_max], color='gray', linestyle='--', alpha=0.5, label='xGF = xGA')
                
                ax.scatter(scatter_stats['xg_for_60'], scatter_stats['xg_against_60'], alpha=0.7)
                
                for _, row in scatter_stats.iterrows():
                    ax.annotate(row['name'], (row['xg_for_60'], row['xg_against_60']),
                                xytext=(5, 5), textcoords='offset points', fontsize=8)
                
                if league_xg_per60 > 0:
                    avg = league_xg_per60
                    # Only plot if within limits
                    if limit_min <= avg <= limit_max:
                        ax.axvline(avg, color='k', linestyle=':', label='League Avg')
                        ax.axhline(avg, color='k', linestyle=':')
                
                ax.set_aspect('equal')
                ax.set_xlabel('xG For / 60')
                ax.set_ylabel('xG Against / 60')
                ax.set_title(f'Player xG Rates: {team or "Selected Players"} ({season})')
                ax.grid(True, alpha=0.3)
                
                scatter_path = os.path.join(out_dir, 'player_scatter.png')
                fig.savefig(scatter_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"players: Saved scatter plot to {scatter_path}")
        except Exception as e:
            print(f"players: Error creating scatter plot: {e}")

    print("players: Analysis complete.")
    return player_stats

def league(season: str = '20252026',
           condition: Optional[dict] = None,
           mode: str = 'compute',
           baseline_path: Optional[str] = None,
           teams: Optional[list] = None):
    """
    Compute or load league-wide xG baseline heatmaps and summary stats.
    
    Args:
        season: Season string.
        condition: Filtering condition (e.g., {'game_state': ['5v5']}).
        mode: 'compute' to calculate, 'load' to read from disk.
        baseline_path: Custom path for baseline files.
        teams: Optional list of teams to process (if None, processes all).
        
    Returns:
        dict: Baseline data and summary stats.
    """
    import os
    import json
    import numpy as np
    import pandas as pd
    from . import timing

    # 1. Setup Paths
    if baseline_path is None:
        baseline_path = _resolve_baseline_path(season, condition)
    os.makedirs(baseline_path, exist_ok=True)
    
    baseline_npy = os.path.join(baseline_path, f'{season}_league_baseline.npy')
    baseline_right_npy = os.path.join(baseline_path, f'{season}_league_baseline_right.npy')
    baseline_json = os.path.join(baseline_path, f'{season}_league_baseline.json')
    summary_json = os.path.join(baseline_path, f'{season}_team_summary.json')
    summary_csv = os.path.join(baseline_path, f'{season}_team_summary.csv')
    intermediates_dir = os.path.join(baseline_path, 'intermediates')
    os.makedirs(intermediates_dir, exist_ok=True)

    # 2. Load Mode
    if mode == 'load':
        if os.path.exists(baseline_npy) and os.path.exists(baseline_json):
            try:
                print(f"league: loading baseline from {baseline_npy}")
                combined_norm = np.load(baseline_npy)
                combined_norm_right = None
                if os.path.exists(baseline_right_npy):
                    combined_norm_right = np.load(baseline_right_npy)
                
                with open(baseline_json, 'r') as f:
                    b_stats = json.load(f)
                
                summary_data = []
                if os.path.exists(summary_json):
                    with open(summary_json, 'r') as f:
                        summary_data = json.load(f)
                        
                return {
                    'combined_norm': combined_norm,
                    'combined_norm_right': combined_norm_right,
                    'baseline_stats': b_stats,
                    'summary': summary_data,
                    'intermediates_dir': intermediates_dir
                }
            except Exception as e:
                print(f"league: failed to load baseline: {e}. Recomputing...")
        else:
            print("league: baseline not found. Recomputing...")

    # 3. Compute Mode
    print(f"league: computing baseline for {season}...")
    
    # Load Data
    df_season = timing.load_season_df(season)
    if df_season is None or df_season.empty:
        print("league: No season data found.")
        return {}
        
    print("league: Predicting xG...")
    try:
        csv_path = locate_season_csv(season)
    except Exception:
        csv_path = None
    df_season, _, _ = _predict_xgs(df_season, csv_path=csv_path)
    
    # Load Teams
    if teams is None:
        teams_path = os.path.join('web', 'static', 'teams.json')
        try:
            with open(teams_path, 'r') as f:
                teams_data = json.load(f)
            team_list = [t.get('abbr') for t in teams_data if 'abbr' in t]
        except Exception:
            team_list = sorted(df_season['home_abb'].dropna().unique().tolist())
    else:
        team_list = list(teams)
        
    print(f"league: Processing {len(team_list)} teams...")
    
    # Aggregators
    sum_map_left = None
    sum_map_right = None
    total_seconds_left = 0.0
    total_seconds_right = 0.0
    total_xg_left = 0.0
    
    summary_rows = []
    
    for team in team_list:
        # print(f"league: processing {team}...")
        team_cond = condition.copy() if condition else {}
        team_cond['team'] = team
        
        try:
            # Calculate robust timing first
            timing_res = timing.compute_game_timing(df_season, team_cond, season=season)
            t_seconds = timing_res.get('aggregate', {}).get('intersection_seconds_total', 0.0)
            
            # Call xgs_map with heatmap_only=True for efficiency
            # We need raw heatmaps (counts), not normalized yet
            # xgs_map returns: out_path, heatmaps, df, stats
            _, heatmaps, _, stats = xgs_map(
                season=season,
                data_df=df_season,
                condition=team_cond,
                return_heatmaps=True,
                heatmap_only=True, # Skip plotting
                total_seconds=t_seconds, # Pass robust timing
                show=False
            )
            
            if not heatmaps or not stats:
                continue
                
            # Extract maps
            tm = heatmaps.get('team')
            om = heatmaps.get('other')
            sec = stats.get('team_seconds', 0.0)
            
            # Save intermediate
            np.savez_compressed(
                os.path.join(intermediates_dir, f'{team}_maps.npz'),
                team_map=tm if tm is not None else np.array([]),
                other_map=om if om is not None else np.array([]),
                stats=stats
            )
            
            # Aggregate
            if tm is not None:
                if sum_map_left is None: sum_map_left = np.zeros_like(tm, dtype=float)
                sum_map_left += tm
                total_xg_left += np.nansum(tm)
                total_seconds_left += sec
                
            if om is not None:
                if sum_map_right is None: sum_map_right = np.zeros_like(om, dtype=float)
                sum_map_right += om
                total_seconds_right += sec
                
            # Summary Stats
            row = stats.copy()
            row['team'] = team
            summary_rows.append(row)
            
        except Exception as e:
            print(f"league: error processing {team}: {e}")
            continue
            
    # 4. Finalize Baseline
    league_baseline_map = None
    league_baseline_right = None
    
    if total_seconds_left > 0 and sum_map_left is not None:
        league_baseline_map = (sum_map_left / total_seconds_left) * 3600.0
    else:
        league_baseline_map = np.zeros((86, 201))
        
    if total_seconds_right > 0 and sum_map_right is not None:
        league_baseline_right = (sum_map_right / total_seconds_right) * 3600.0
        
    # 5. Save Outputs
    np.save(baseline_npy, league_baseline_map)
    if league_baseline_right is not None:
        np.save(baseline_right_npy, league_baseline_right)
        
    b_stats = {
        'season': season,
        'total_left_seconds': total_seconds_left,
        'total_left_xg': total_xg_left,
        'xg_per60': float(np.nansum(league_baseline_map)),
        'xg_per60_right': float(np.nansum(league_baseline_right)) if league_baseline_right is not None else 0.0
    }
    
    with open(baseline_json, 'w') as f:
        json.dump(b_stats, f, indent=2)
        
    # Merge with existing summary if needed
    if teams is not None and os.path.exists(summary_json):
        try:
            with open(summary_json, 'r') as f:
                existing_summary = json.load(f)
            
            # Convert to dict for easy merging by team name
            summary_dict = {s['team']: s for s in existing_summary if 'team' in s and s['team'] != 'League'}
            
            # Update with new rows
            for row in summary_rows:
                if 'team' in row:
                    summary_dict[row['team']] = row
            
            # Convert back to list
            summary_rows = list(summary_dict.values())
            print(f"league: Merged {len(summary_rows)} teams into summary (updated {len(teams)} teams).")
        except Exception as e:
            print(f"league: Warning: failed to merge summary: {e}")

    # Calculate League Totals for Summary
    if summary_rows:
        # Filter out any existing League row just in case (though we filtered above)
        team_rows = [r for r in summary_rows if r.get('team') != 'League']
        
        l_goals = sum(r.get('team_goals', 0) for r in team_rows)
        l_xgs = sum(r.get('team_xgs', 0.0) for r in team_rows)
        l_attempts = sum(r.get('team_attempts', 0) for r in team_rows)
        l_seconds = sum(r.get('team_seconds', 0.0) for r in team_rows)
        l_xg_per60 = (l_xgs / l_seconds * 3600.0) if l_seconds > 0 else 0.0
        
        league_row = {
            'team': 'League',
            'team_goals': l_goals,
            'team_xgs': l_xgs,
            'team_attempts': l_attempts,
            'team_seconds': l_seconds,
            'team_xg_per60': l_xg_per60,
            'n_games': sum(r.get('n_games', 0) for r in team_rows), # Rough sum
            'other_goals': sum(r.get('other_goals', 0) for r in team_rows),
            'other_xgs': sum(r.get('other_xgs', 0.0) for r in team_rows),
            'other_attempts': sum(r.get('other_attempts', 0) for r in team_rows),
            'other_seconds': l_seconds, # Same as team seconds usually
            'other_xg_per60': (sum(r.get('other_xgs', 0.0) for r in team_rows) / l_seconds * 3600.0) if l_seconds > 0 else 0.0
        }
        summary_rows = team_rows + [league_row]

    with open(summary_json, 'w') as f:
        json.dump(summary_rows, f, indent=2)
        
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
        
    print(f"league: Baseline saved to {baseline_path}")
    
    return {
        'combined_norm': league_baseline_map,
        'combined_norm_right': league_baseline_right,
        'baseline_stats': b_stats,
        'summary': summary_rows,
        'intermediates_dir': intermediates_dir
    }



def calculate_shot_attempts(df: pd.DataFrame, team_val: str) -> dict:
    """
    Calculate shot attempts (Corsi) stats from the filtered dataframe.
    
    Args:
        df: Filtered DataFrame containing event data.
        team_val: Team identifier (abbreviation or ID) to filter for.
        
    Returns:
        dict: Dictionary containing:
            - home_attempts: int (Team shot attempts)
            - away_attempts: int (Opponent shot attempts)
            - home_shot_pct: float (Team CF%)
            - away_shot_pct: float (Opponent CF%)
    """
    stats = {
        'home_attempts': 0,
        'away_attempts': 0,
        'home_shot_pct': 0.0,
        'away_shot_pct': 0.0
    }
    
    if df is None or df.empty:
        return stats
        
    # Filter for shot attempts
    attempts_df = df[df['event'].astype(str).str.strip().str.lower().isin(['shot-on-goal', 'missed-shot', 'blocked-shot'])]
    
    if attempts_df.empty:
        return stats
        
    # Identify team rows
    def _is_team_row(r):
        try:
            # Check if team_id matches team (which is an abbr here)
            # We need to match abbr to home_abb/away_abb to find team_id
            tupper = str(team_val).upper()
            home_abb = r.get('home_abb')
            away_abb = r.get('away_abb')
            if home_abb is not None and str(home_abb).upper() == tupper:
                return str(r.get('team_id')) == str(r.get('home_id'))
            if away_abb is not None and str(away_abb).upper() == tupper:
                return str(r.get('team_id')) == str(r.get('away_id'))
            return False
        except Exception:
            return False

    is_team_mask = attempts_df.apply(_is_team_row, axis=1)
    team_attempts = int(is_team_mask.sum())
    opp_attempts = int((~is_team_mask).sum())
    
    stats['home_attempts'] = team_attempts
    stats['away_attempts'] = opp_attempts
    
    total_attempts = team_attempts + opp_attempts
    if total_attempts > 0:
        stats['home_shot_pct'] = 100.0 * team_attempts / total_attempts
        stats['away_shot_pct'] = 100.0 * opp_attempts / total_attempts
        
    return stats


def compute_relative_map(team_map, league_baseline_left, team_seconds, other_map, other_seconds, baseline_threshold=1e-6, league_baseline_right=None, metric='diff'):
    """
    Compute the relative xG map (Team vs League and Defense vs League).
    
    Args:
        team_map: Raw team heatmap (offense).
        league_baseline_left: League baseline heatmap (offense, left-oriented).
        team_seconds: Total seconds for team normalization.
        other_map: Raw other heatmap (defense).
        other_seconds: Total seconds for other normalization.
        baseline_threshold: Threshold to avoid division by zero (used for pct metric).
        league_baseline_right: Optional right-oriented baseline.
        metric: 'diff' for (Team - League), 'pct' for (Team - League) / League.
        
    Returns:
        tuple: (combined_rel_map, rel_off_pct, rel_def_pct, relative_off_per60, relative_def_per60)
    """
    import numpy as np
    
    # 1. Normalization to xGs/60
    if team_seconds <= 0: team_seconds = 1.0
    if other_seconds <= 0: other_seconds = 1.0

    # Normalize team map (Offense, Left)
    team_map_norm = np.asarray(team_map, dtype=float) / team_seconds * 3600.0
    
    # Normalize other map (Defense, Right)
    if other_map is not None:
        other_map_norm = np.asarray(other_map, dtype=float) / other_seconds * 3600.0
    else:
        other_map_norm = np.zeros_like(team_map_norm)

    # 2. League Baselines
    # league_baseline_left is already Left-oriented (Offense)
    
    # Create Right-oriented baseline (Defense)
    # If explicit right baseline provided, use it. Else flip left baseline (assuming symmetry).
    if league_baseline_right is not None:
        # Ensure it's right-oriented (it should be if coming from league())
        league_baseline_right_use = league_baseline_right
    else:
        # Fallback: Flip Left baseline
        league_baseline_right_use = np.fliplr(league_baseline_left)

    # 3. Compute Relative Map
    
    # Grid setup
    gx = np.arange(-100.0, 100.0 + 1.0, 1.0)
    gy = np.arange(-42.5, 42.5 + 1.0, 1.0)
    XX, YY = np.meshgrid(gx, gy)
    
    # Masks
    # Left Zone: x <= -24.0 (Offense)
    # Right Zone: x >= 24.0 (Defense)
    # Neutral Zone: -24.0 < x < 24.0 (Masked)
    mask_left = (XX <= -24.0)
    mask_right = (XX >= 24.0)
    
    # Initialize combined map with NaNs
    combined_rel_map = np.full_like(team_map_norm, np.nan)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # Left Side: Offense vs League Left
        diff_l = team_map_norm - league_baseline_left
        
        if metric == 'pct':
            denom_l = np.maximum(league_baseline_left, baseline_threshold)
            rel_l = diff_l / denom_l
            has_signal_l = (team_map_norm > baseline_threshold) | (league_baseline_left > baseline_threshold)
        else:
            # metric == 'diff'
            rel_l = diff_l
            # For diff, we don't necessarily need a signal threshold, but keeping it clean for zero areas might be good.
            # However, diff is valid everywhere. Let's just use mask_left.
            has_signal_l = np.ones_like(diff_l, dtype=bool) 
        
        valid_l = mask_left & has_signal_l
        combined_rel_map[valid_l] = rel_l[valid_l]
        
        # Right Side: Defense vs League Right
        diff_r = other_map_norm - league_baseline_right_use
        
        if metric == 'pct':
            denom_r = np.maximum(league_baseline_right_use, baseline_threshold)
            rel_r = diff_r / denom_r
            has_signal_r = (other_map_norm > baseline_threshold) | (league_baseline_right_use > baseline_threshold)
        else:
            # metric == 'diff'
            rel_r = diff_r
            has_signal_r = np.ones_like(diff_r, dtype=bool)
            
        valid_r = mask_right & has_signal_r
        combined_rel_map[valid_r] = rel_r[valid_r]

    # Calculate aggregate relative stats (always xG/60 diff and %)
    team_xg_per60 = np.nansum(team_map_norm)
    other_xg_per60 = np.nansum(other_map_norm)
    league_avg_xg_per60 = np.nansum(league_baseline_left)
    league_avg_xg_per60_right = np.nansum(league_baseline_right_use)
    
    relative_off_per60 = team_xg_per60 - league_avg_xg_per60
    relative_def_per60 = other_xg_per60 - league_avg_xg_per60_right
    
    rel_off_pct = 0.0
    rel_def_pct = 0.0
    if league_avg_xg_per60 > 1e-6:
        rel_off_pct = (relative_off_per60 / league_avg_xg_per60) * 100.0
        
    if league_avg_xg_per60_right > 1e-6:
        rel_def_pct = (relative_def_per60 / league_avg_xg_per60_right) * 100.0
        
    return combined_rel_map, rel_off_pct, rel_def_pct, relative_off_per60, relative_def_per60


def _predict_xgs(df_filtered: pd.DataFrame, model_path='analysis/xgs/xg_model_nested.joblib', behavior='load', csv_path=None):
    """Load/train classifier if needed and predict xgs for df rows; returns (df_with_xgs, clf, meta).

    Meta is (final_feature_names, categorical_levels_map) to be reused by callers.
    """
    import pandas as pd
    import numpy as np
    from . import fit_xgs

    df = df_filtered
    if df.shape[0] == 0:
        return df, None, None

    need_predict = ('xgs' not in df.columns) or (df['xgs'].isna().all()) or (behavior == 'overwrite')
    if not need_predict:
        return df, None, None

    # get classifier (respect behavior, fallback to train on failure)
    # Map 'overwrite' to 'load' for the model loader
    clf_behavior = 'load' if behavior == 'overwrite' else behavior
    
    try:
        # prefer explicit csv_path if provided; otherwise pass None when using data_df
        # Use csv_path passed in
        clf, feature_names, cat_levels = fit_xgs.get_clf(model_path, clf_behavior, csv_path=csv_path, model_type='nested')
    except Exception as e:
        print(f"xgs_map: get_clf failed with {e}")
        if csv_path:
             print("...trying to train a new model")
             clf, feature_names, cat_levels = fit_xgs.get_clf(model_path, 'train', csv_path=csv_path)
        else:
             print("...cannot train new model without csv_path. Returning empty.")
             return df, None, None

    # Check if this is the Nested Model
    # We can check type safely
    is_nested = False
    try:
        from .fit_nested_xgs import NestedXGClassifier
        if isinstance(clf, NestedXGClassifier):
            is_nested = True
    except ImportError:
        pass
    
    # Also check via string just in case of reload/import issues
    if not is_nested and type(clf).__name__ == 'NestedXGClassifier':
        is_nested = True

    if is_nested:
        # NESTED MODEL LOGIC:
        # 1. Apply Imputation (Crucial! Otherwise blocked shots look like close-range shots)
        try:
            from . import impute
            # Use same method as training
            df_model = impute.impute_blocked_shot_origins(df.copy(), method='mean_6')
        except ImportError:
            # Fallback if impute not found relative (shouldn't happen)
            import impute
            df_model = impute.impute_blocked_shot_origins(df.copy(), method='mean_6')
        except Exception as e:
            print(f"Warning: Imputation failed in _predict_xgs: {e}")
            df_model = df.copy()

        # 2. Features
        features = ['distance', 'angle_deg', 'game_state', 'is_net_empty', 'shot_type'] # Standard Nested Features
        
        # Ensure features exist
        for f in features:
            if f not in df_model.columns:
                if f == 'shot_type':
                    df_model[f] = 'Unknown'
                elif f == 'game_state':
                    df_model[f] = '5v5'
                else:
                    df_model[f] = 0
            else:
                # Basic fillna
                if f == 'shot_type':
                    df_model[f] = df_model[f].fillna('Unknown')
                elif f == 'game_state':
                    df_model[f] = df_model[f].fillna('5v5')
                else:
                     df_model[f] = df_model[f].fillna(0)
                     
        # Metadata pass-through
        final_features = features
        cat_levels = None 

    else:
        # STANDARD MODEL LOGIC:
        # Prepare the model DataFrame using canonical feature list used for cleaning/generation
        # We must NOT use 'feature_names' (from meta) as input features, because meta features are 
        # usually transformed (e.g. game_state_code) while input DF has raw features (game_state).
        
        input_features = ['distance', 'angle_deg', 'game_state', 'is_net_empty', 'shot_type']
        
        # We pass cat_levels to ensure consistent encoding
        df_model, final_feature_cols_game, cat_map_game = fit_xgs.clean_df_for_model(df.copy(), input_features, fixed_categorical_levels=cat_levels)
    
        # If the model expects specific features (from meta), ensure we have them
        if feature_names:
            missing = [f for f in feature_names if f not in df_model.columns]
            if missing:
                # If missing, it might be because input_features didn't include them?
                # or encoding didn't produce them?
                print(f"Warning: Model expects features {missing} which were not produced by cleaning. Using available features.")
                # We can't fix this easily if logic differs. 
                # But for now, we assume standard pipeline.
                pass
            
            # Filter/Order columns to match classifier expectation
            valid_feats = [f for f in feature_names if f in df_model.columns]
            df_model = df_model[valid_feats]
            final_features = valid_feats
        else:
             final_features = final_feature_cols_game

    # ensure xgs column exists
    df['xgs'] = np.nan
    
    # FILTER: Only predict for valid shot events
    valid_events = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal']
    # Create a mask for valid rows in the ORIGINAL dataframe
    # We must use df index to align with df_model
    mask_valid = df['event'].isin(valid_events)
    
    # We only want to predict for rows that are both in df_model AND are valid events.
    # df_model generally preserves index of df (or subset if clean_df dropped rows).
    valid_indices = df_model.index.intersection(df.index[mask_valid])
    
    # predict probabilities when possible
    # PREDICTION
    if clf is not None and not valid_indices.empty:
        try:
            target_df = df_model.loc[valid_indices]
            
            if target_df.shape[0] > 0:
                 probs = clf.predict_proba(target_df)[:, 1]
                 # Assignment: Ensure index alignment! 
                 df.loc[valid_indices, 'xgs'] = probs
            
        except Exception as e:
            print(f"xgs_map: prediction failed with {e}")
            pass
    
    # Fill remaining (invalid events, or failed predictions) with 0.0
    df['xgs'] = df['xgs'].fillna(0.0)

    # SAFEGUARD: Explicitly force 0.0 for known non-shooting events, just in case
    # mask_valid was the "allowed" list. Anything NOT in valid_events should be 0.
    # The fillna(0.0) above handles NaN, but if prediction somehow ran on invalid rows earlier,
    # this overwrites them.
    df.loc[~mask_valid, 'xgs'] = 0.0

    return df, clf, (final_features, cat_levels)


def generate_condition_plots(season, condition_name, out_dir):
    """
    Generate condition-level plots (e.g. scatter plots) from saved summary data.
    Reads [season]_team_summary.json from out_dir.
    """
    import json
    import os
    
    summary_json_path = os.path.join(out_dir, f'{season}_team_summary.json')
    if not os.path.exists(summary_json_path):
        print(f"generate_condition_plots: summary file not found at {summary_json_path}")
        return

    try:
        with open(summary_json_path, 'r') as f:
            summary_list = json.load(f)
        
        if not summary_list:
            print(f"generate_condition_plots: summary list is empty in {summary_json_path}")
            return

        # Generate scatter plot
        generate_scatter_plot(summary_list, out_dir, condition_name=condition_name)
        
        # Generate relative maps (using updated summary data with percentiles)
        generate_relative_maps(season, condition_name, out_dir, summary_list)
        
    except Exception as e:
        print(f"generate_condition_plots: error generating plots for {condition_name}: {e}")

def generate_relative_maps(season, condition_name, out_dir, summary_data):
    """
    Generate relative map PNGs for each team using the provided summary data.
    This allows maps to be generated after percentiles are calculated for the whole league.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from .rink import draw_rink
    from .plot import add_summary_text
    import matplotlib.ticker as mticker
    
    # Pre-calculate league baseline range or use fixed?
    # season() used fixed logic for 5v5.
    
    for row in summary_data:
        team = row.get('team')
        if not team: continue
        
        try:
            # Load relative map data
            rel_map_npy = os.path.join(out_dir, f'{team}_relative_combined.npy')
            if not os.path.exists(rel_map_npy):
                # print(f"generate_relative_maps: npy not found for {team} at {rel_map_npy}")
                continue
                
            combined_rel_map = np.load(rel_map_npy)
            
            # Plotting logic (copied/adapted from season loop)
            fig, ax = plt.subplots(figsize=(10, 5))
            draw_rink(ax=ax)
            
            # Determine vmin/vmax
            # For diff metric, use SymLogNorm to show low-density structure.
            # 50th %ile is ~1e-6, 80th is ~1.7e-5. Max is ~1e-3.
            # linthresh=1e-5 seems appropriate.
            
            from matplotlib.colors import SymLogNorm
            norm = SymLogNorm(linthresh=1e-5, linscale=1.0, vmin=-0.0006, vmax=0.0006, base=10)
            
            # Custom locking for ALL conditions
            # Ticks for log scale: +/- 1e-5, 1e-4, 6e-4
            # We want human readable labels
            cbar_ticks = [-0.0006, -0.0001, -0.00001, 0, 0.00001, 0.0001, 0.0006]
            cbar_ticklabels = ['High -', 'Med -', 'Low -', 'Avg', 'Low +', 'Med +', 'High +']

            # Grid extent (hardcoded or passed? season() had gx, gy)
            # We need gx, gy to define extent.
            # They are fixed for the standard grid.
            # Let's assume standard grid params: 100, 85, res=1.0?
            # Actually, compute_xg_heatmap_from_df returns gx, gy.
            # We can re-create them easily if we know the grid params.
            # Default: x_range=(-100, 100), y_range=(-42.5, 42.5), res=1.0
            # Let's verify grid params or save them?
            # Saving them in npy would be better, but for now let's assume standard.
            # Or we can just use the shape of combined_rel_map to infer?
            # shape is (ny, nx).
            # If res=1.0, nx=201, ny=86.
            
            ny, nx = combined_rel_map.shape
            # Assuming standard grid centered at 0
            # x from -100 to 100 (201 points)
            # y from -42.5 to 42.5 (86 points)
            
            # If shape matches standard, use standard extent.
            if nx == 201 and ny == 86:
                extent = (-100.5, 100.5, -43.0, 43.0) # boundaries for 201x86 cells centered on integer coords?
                # Wait, compute_xg_heatmap_from_df:
                # x_range = (-100, 100), y_range = (-42.5, 42.5)
                # grid_x = np.arange(x_range[0], x_range[1] + grid_res, grid_res) -> -100 to 100 inclusive (201)
                # extent = (gx[0] - res/2, gx[-1] + res/2, ...)
                # -100 - 0.5 = -100.5
                # 100 + 0.5 = 100.5
                extent = (-100.5, 100.5, -43.0, 43.0)
            else:
                # Fallback or generic
                extent = (-100, 100, -42.5, 42.5)

            cmap = plt.get_cmap('RdBu_r')
            try:
                cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))
            except:
                pass
            
            m = np.ma.masked_invalid(combined_rel_map)
            im = ax.imshow(m, extent=extent, origin='lower', cmap=cmap, norm=norm)
            
            # Text Stats
            text_stats = row.copy()
            # Map keys for add_summary_text
            text_stats['home_xg'] = text_stats.get('team_xgs', 0.0)
            text_stats['away_xg'] = text_stats.get('other_xgs', 0.0)
            text_stats['have_xg'] = True
            text_stats['home_goals'] = text_stats.get('team_goals', 0)
            text_stats['away_goals'] = text_stats.get('other_goals', 0)
            text_stats['home_attempts'] = text_stats.get('team_attempts', 0)
            text_stats['away_attempts'] = text_stats.get('other_attempts', 0)
            
            # Add summary text
            add_summary_text(
                ax=ax,
                stats=text_stats,
                main_title=f"{team} {condition_name} Relative to League",
                is_season_summary=True,
                team_name=team,
                filter_str=condition_name
            )
            
            # Colorbar
            cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
            cbar.set_label('Relative xG/60 Difference', rotation=270, labelpad=20)
            if cbar_ticks:
                cbar.set_ticks(cbar_ticks)
            if cbar_ticklabels:
                cbar.ax.set_yticklabels(cbar_ticklabels)
            # cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0)) # No longer percent
            
            relative_map_path = os.path.join(out_dir, f'{team}_relative_map.png')
            fig.savefig(relative_map_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # print(f"generate_relative_maps: saved {relative_map_path}")
            
        except Exception as e:
            print(f"generate_relative_maps: error for {team}: {e}")


def generate_scatter_plot(summary_list, out_dir, condition_name=""):
    """Generate xGF/60 vs xGA/60 scatter plot with logos."""
    if not summary_list:
        return

    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    import urllib.request
    from PIL import Image
    import io
    import os

    # Use Agg backend if not already set
    
    df = pd.DataFrame(summary_list)
    
    # Map columns if needed (season analysis uses different keys than players)
    # season: relative_off_per60, relative_def_per60, team_xg_per60, other_xg_per60
    if 'team_xg_per60' in df.columns and 'other_xg_per60' in df.columns:
        df['xGF/60'] = df['team_xg_per60']
        df['xGA/60'] = df['other_xg_per60']
        df['Team'] = df['team']
    
    # Check if required columns exist
    if 'xGF/60' not in df.columns or 'xGA/60' not in df.columns:
        print("Missing xGF/60 or xGA/60 columns for scatter plot")
        return

    fig, ax = plt.subplots(figsize=(10, 10)) # Square figure
    
    # Helper to fetch logo
    def get_team_logo(team_abbr):
        try:
            url = f"https://assets.nhle.com/logos/nhl/svg/{team_abbr}_light.svg"
            # SVG support in matplotlib is tricky, try PNG from another source or convert?
            # Actually, NHL API provides SVGs. Matplotlib doesn't natively read SVG into OffsetImage easily without conversion.
            # Let's use a PNG source or fallback to text if SVG fails.
            # Alternative: https://assets.nhle.com/logos/nhl/svg/{team}_light.svg is standard but SVG.
            # Try getting PNG from ESPN or similar? Or just use text if SVG is hard.
            # Wait, user asked for logos.
            # Let's try to find a PNG source.
            # https://a.espncdn.com/i/teamlogos/nhl/500/{team}.png
            # Mapping might be needed for some teams (UTA?).
            
            # Simple mapping for ESPN
            espn_map = {
                'UTA': 'utah', # Guessing for Utah
                'VGK': 'vgs', # ESPN uses vgs? No, vgk usually.
                # Let's try standard abbr first.
            }
            q_team = espn_map.get(team_abbr, team_abbr)
            url = f"https://a.espncdn.com/i/teamlogos/nhl/500/{q_team}.png"
            
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                data = response.read()
            
            im = Image.open(io.BytesIO(data))
            return im
        except Exception as e:
            # print(f"Failed to fetch logo for {team_abbr}: {e}")
            return None

    # Plot points/logos
    # Calculate limits to ensure square aspect ratio and unity line visibility
    max_val = max(df['xGF/60'].max(), df['xGA/60'].max())
    min_val = min(df['xGF/60'].min(), df['xGA/60'].min())
    padding = (max_val - min_val) * 0.1
    limit_max = max_val + padding
    limit_min = max(0, min_val - padding)
    
    # Set limits explicitly to be equal
    ax.set_xlim(limit_min, limit_max)
    ax.set_ylim(limit_min, limit_max)
    
    # Invert Y axis (lower xGA is better, usually at top? User asked to "invert the y-axis")
    # Standard plot: Y is xGA. Inverted means 0 at top.
    ax.invert_yaxis()
    
    # Unity line (x=y)
    ax.plot([limit_min, limit_max], [limit_min, limit_max], color='gray', linestyle='--', alpha=0.5, label='xGF = xGA')

    # Plot logos
    for i, row in df.iterrows():
        team = row['Team']
        x = row['xGF/60']
        y = row['xGA/60']
        
        logo_img = get_team_logo(team)
        if logo_img:
            imagebox = OffsetImage(logo_img, zoom=0.05) # Adjust zoom as needed
            ab = AnnotationBbox(imagebox, (x, y), frameon=False)
            ax.add_artist(ab)
        else:
            ax.scatter(x, y, alpha=0.7)
            ax.annotate(team, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax.set_xlabel('xGF/60')
    ax.set_ylabel('xGA/60')
    
    title = condition_name if condition_name else 'Team xG Rates'
    ax.set_title(title)
    
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Remove legend as requested (dotted line key)
    # ax.legend() 
    
    # Save
    out_path = os.path.join(out_dir, 'scatter.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved scatter plot to {out_path}")


def season(season: str = '20252026',
           team: str = None,
           condition: Optional[dict] = None,
           league_data: Optional[dict] = None,
           out_dir: Optional[str] = None):
    """
    Analyze a single team for a given condition.
    Generates xG map and Relative xG map.
    
    Args:
        season: Season string.
        team: Team abbreviation.
        condition: Filtering condition.
        league_data: Optional pre-loaded league data (from analyze.league).
        out_dir: Output directory.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from .plot import add_summary_text, rink_goal_xs
    from .rink import draw_rink
    # Import timing module
    import sys
    try:
        from . import timing
    except ImportError:
        timing = None
    import json

    if team is None:
        print("season: Team must be specified.")
        return

    # 1. Setup Paths
    if out_dir is None:
        # Default to static/league/{season}/{cond_name}
        base_path = _resolve_baseline_path(season, condition)
        out_dir = base_path
    os.makedirs(out_dir, exist_ok=True)
    
    # 2. Load League Data
    if league_data is None:
        # Try to load
        league_data = league(season=season, condition=condition, mode='load')
        
    if not league_data or league_data.get('combined_norm') is None:
        print(f"season: League baseline not found for {season}. Cannot generate relative maps.")
        return

    league_map = league_data['combined_norm']
    league_map_right = league_data['combined_norm_right']
    intermediates_dir = league_data.get('intermediates_dir')
    
    # 3. Get Team Data
    # Try to load from intermediates
    team_map = None
    other_map = None
    stats = {}
    
    loaded_intermediate = False
    if intermediates_dir:
        int_path = os.path.join(intermediates_dir, f'{team}_maps.npz')
        if os.path.exists(int_path):
            try:
                data = np.load(int_path, allow_pickle=True)
                team_map = data['team_map']
                other_map = data['other_map']
                stats = data['stats'].item()
                loaded_intermediate = True
                # Handle 0-d arrays if empty
                if team_map.ndim == 0: team_map = None
                if other_map.ndim == 0: other_map = None
            except Exception as e:
                print(f"season: failed to load intermediate for {team}: {e}")
    
    if not loaded_intermediate:
        print(f"season: Processing {team} (Game-Centric)...")
        
        # Load season data once
        df_season = timing.load_season_df(season)
        df_season, _, _ = _predict_xgs(df_season)
        
        # Identify games for this team
        team_games_df = df_season[
            (df_season['home_abb'] == team) | 
            (df_season['away_abb'] == team)
        ]
        game_ids = sorted(team_games_df['game_id'].unique())
        print(f"season: Found {len(game_ids)} games for {team}")
        
        cache_dir = os.path.join('data', season, 'game_stats_team')
        os.makedirs(cache_dir, exist_ok=True)
        
        total_team_map = None
        total_other_map = None
        total_stats = {}
        
        for i, game_id in enumerate(game_ids):
            cache_file = os.path.join(cache_dir, f"{game_id}_{team}.pkl")
            
            game_res = None
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        game_res = pickle.load(f)
                except Exception: pass
                
            if game_res is None:
                # Calculate
                g_cond = condition.copy() if condition else {}
                g_cond['team'] = team
                
                df_game = df_season[df_season['game_id'] == game_id]
                
                # Calculate total seconds using shared cache
                g_intervals = timing.get_game_intervals_cached(game_id, season, g_cond)
                g_seconds = sum(e - s for s, e in g_intervals)
                
                _, heatmaps, _, g_stats = xgs_map(
                    season=season,
                    data_df=df_game,
                    condition=g_cond,
                    return_heatmaps=True,
                    heatmap_only=True,
                    show=False,
                    total_seconds=g_seconds,
                    use_intervals=True,
                    intervals_input={'per_game': {game_id: {'intersection_intervals': g_intervals}}}
                )
                
                tm = heatmaps.get('team') if heatmaps else None
                om = heatmaps.get('other') if heatmaps else None
                
                game_res = {
                    'team_map': tm,
                    'other_map': om,
                    'stats': g_stats
                }
                
                # Save to cache
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(game_res, f)
                except Exception: pass
            
            # Aggregate
            tm = game_res.get('team_map')
            om = game_res.get('other_map')
            g_stats = game_res.get('stats', {})
            
            if tm is not None:
                if total_team_map is None:
                    total_team_map = np.zeros_like(tm)
                    total_other_map = np.zeros_like(om) if om is not None else None
                
                total_team_map += np.nan_to_num(tm)
                if total_other_map is not None and om is not None:
                    total_other_map += np.nan_to_num(om)
            
            # Sum stats
            for k, v in g_stats.items():
                if isinstance(v, (int, float)):
                    total_stats[k] = total_stats.get(k, 0) + v
                    
        team_map = total_team_map
        other_map = total_other_map
        stats = total_stats

    if team_map is None:
        print(f"season: No data for {team}")
        return

    seconds = stats.get('team_seconds', 0.0)
    if seconds <= 0:
        print(f"season: Zero seconds for {team}")
        return

    # 4. Calculate Percentiles
    # We need the distribution of xG/60 across the league for this condition
    # league_data['summary'] contains list of team stats
    summary_list = league_data.get('summary', [])
    
    # If summary_list is small (e.g. subset run), try to load full summary from disk
    if len(summary_list) < 20:
        try:
            # Resolve path to summary json
            # We know out_dir is static/league/{season}/{cond} usually
            # But we can reconstruct it safely
            base_path = _resolve_baseline_path(season, condition)
            full_summary_path = os.path.join(base_path, f'{season}_team_summary.json')
            if os.path.exists(full_summary_path):
                with open(full_summary_path, 'r') as f:
                    full_summary = json.load(f)
                if len(full_summary) > len(summary_list):
                    print(f"season: Loaded full league summary ({len(full_summary)} teams) for percentiles.")
                    summary_list = full_summary
        except Exception as e:
            print(f"season: Warning: failed to load full summary for percentiles: {e}")

    off_percentile = None
    def_percentile = None
    
    if summary_list:
        # Extract distributions
        off_vals = []
        def_vals = []
        for s in summary_list:
            if s.get('team') == 'League': continue
            if s.get('team_xg_per60') is not None: off_vals.append(s['team_xg_per60'])
            if s.get('other_xg_per60') is not None: def_vals.append(s['other_xg_per60']) # Note: other_xg_per60 is defense
            
        t_off = stats.get('team_xg_per60')
        t_def = stats.get('other_xg_per60') # Defense metric
        
        from scipy import stats as sp_stats
        if t_off is not None and off_vals:
            off_percentile = sp_stats.percentileofscore(off_vals, t_off)
        if t_def is not None and def_vals:
            # For defense, lower is better? Usually percentile implies "better than X%".
            # If lower xGA is better, then we want percentile of (1/x) or similar?
            # Or just raw percentile: "80th percentile" means high xGA (bad defense).
            # Usually "Defensive Percentile" in hockey viz: 100 = Best Defense (Lowest xGA).
            # So we invert.
            raw_pct = sp_stats.percentileofscore(def_vals, t_def)
            def_percentile = 100.0 - raw_pct
            
    # Update stats with percentiles
    stats['off_percentile'] = off_percentile
    stats['def_percentile'] = def_percentile
    
    # Map team stats to plot-expected keys (home=team, away=other)
    stats['home_goals'] = stats.get('team_goals', 0)
    stats['away_goals'] = stats.get('other_goals', 0)
    stats['home_xg'] = stats.get('team_xgs', 0.0)
    stats['away_xg'] = stats.get('other_xgs', 0.0)
    stats['home_attempts'] = stats.get('team_attempts', 0)
    stats['away_attempts'] = stats.get('other_attempts', 0)
    stats['have_xg'] = True
    
    # Calculate shot percentages (CF%)
    t_att = stats.get('team_attempts', 0)
    o_att = stats.get('other_attempts', 0)
    if t_att + o_att > 0:
        stats['home_shot_pct'] = 100.0 * t_att / (t_att + o_att)
        stats['away_shot_pct'] = 100.0 * o_att / (t_att + o_att)
    else:
        stats['home_shot_pct'] = 0.0
        stats['away_shot_pct'] = 0.0
    
    # 5. Generate Plots
    
    # A. Raw xG Map
    # We need to call plot_events-like logic or just imshow since we have the map.
    # But xgs_map does nice smoothing/binning. We have the binned map.
    # We can just plot the binned map.
    
    # Helper to plot map
    def _plot_map(heatmap, title, filename, is_relative=False, relative_baseline=None, cbar_ticklabels=None):
        fig, ax = plt.subplots(figsize=(10, 5))
        draw_rink(ax=ax)
        
        gx = np.arange(-100.0, 100.0 + 1.0, 1.0)
        gy = np.arange(-42.5, 42.5 + 1.0, 1.0)
        extent = (gx[0] - 0.5, gx[-1] + 0.5, gy[0] - 0.5, gy[-1] + 0.5)
        
        cmap = plt.get_cmap('Reds')
        vmax = None
        vmin = None
        
        if is_relative:
            cmap = plt.get_cmap('RdBu_r')
            
            # Use SymLogNorm for diff metric to enhance contrast
            from matplotlib.colors import SymLogNorm
            # Default vmax
            vmax = 0.0006
            # Fixed scale for ALL conditions to ensure consistency
            cbar_ticks = [-0.0006, -0.0001, -0.00001, 0, 0.00001, 0.0001, 0.0006]
            cbar_ticklabels = ['High -', 'Med -', 'Low -', 'Avg', 'Low +', 'Med +', 'High +']
            
            norm = SymLogNorm(linthresh=1e-5, linscale=1.0, vmin=-vmax, vmax=vmax, base=10)
            vmin = None # Handled by norm
            vmax = None # Handled by norm
            
            # Compute relative map
            # We need compute_relative_map logic here if not passed
            m = heatmap # Assumed passed as relative
        else:
            # Raw map: normalize to per 60
            m = (heatmap / seconds) * 3600.0
            # Smooth? The map from xgs_map is raw counts usually?
            # xgs_map returns 'heatmaps'. If it called compute_xg_heatmap_from_df, it returns raw counts?
            # Let's check xgs_map. It calls compute_xg_heatmap_from_df.
            # compute_xg_heatmap_from_df returns histogram (counts).
            # So yes, normalize.
            # Smoothing is done by gaussian_filter in plot_events usually.
            from scipy.ndimage import gaussian_filter
            m = gaussian_filter(m, sigma=1.5)
            
        # Mask
        m = np.ma.masked_invalid(m)
        if is_relative:
             try: cmap.set_bad(color=(1,1,1,0)) 
             except: pass
        
        if is_relative:
            im = ax.imshow(m, extent=extent, origin='lower', cmap=cmap, norm=norm)
        else:
            im = ax.imshow(m, extent=extent, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        
        # Text
        cond_str = str(condition) if condition else ""
        add_summary_text(
            ax=ax,
            stats=stats,
            main_title=title,
            is_season_summary=True,
            team_name=team,
            full_team_name=team,
            filter_str=cond_str
        )
        ax.axis('off')
        
        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        if is_relative:
            cbar.set_label('Relative xG/60 Difference', rotation=270, labelpad=20)
            import matplotlib.ticker as mticker
            # cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        else:
            cbar.set_label('xG per 60', rotation=270, labelpad=20)
            
        path = os.path.join(out_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        # print(f"season: Saved {filename}")

    # Plot Raw
    _plot_map(team_map, f"{team} xG Rates", f"{team}_xg_map.png", is_relative=False)
    
    # Plot Relative
    # Compute relative map
    if league_map is not None:
        combined_rel_map, rel_off_pct, rel_def_pct, _, _ = compute_relative_map(
            team_map, league_map, seconds, other_map, seconds,
            league_baseline_right=league_map_right,
            metric='diff'
        )
        
        # Update stats with relative pcts
        stats['rel_off_pct'] = rel_off_pct
        stats['rel_def_pct'] = rel_def_pct
        
        cbar_ticklabels = ['High -', 'Med -', 'Low -', 'Avg', 'Low +', 'Med +', 'High +']
        
        _plot_map(combined_rel_map, f"{team} Relative xG", f"{team}_relative.png", is_relative=True, cbar_ticklabels=cbar_ticklabels)
        
        # Save combined relative map for Special Teams stitching
        # We need to save it as .npy
        np.save(os.path.join(out_dir, f'{team}_relative_combined.npy'), combined_rel_map)
    
    # Save per-team summary JSON
    # This is useful for the Special Teams plotter which loads individual team summaries?
    # Actually Special Teams plotter loads the BIG summary file.
    # But we can save individual one too.
    with open(os.path.join(out_dir, f'{season}_{team}_summary.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    return stats


def xgs_map(season: Optional[str] = '20252026', *,
            game_id: Optional[str] = None,

            csv_path: Optional[str] = None,
              model_path: str = 'analysis/xgs/xg_model_nested.joblib',
              behavior: str = 'load',
              out_path: str = 'analysis/xgs/xg_map.png',
              orient_all_left: bool = False,
              events_to_plot: Optional[list] = None,
              show: bool = False,
              return_heatmaps: bool = True,
              # when True, return the filtered dataframe used to create the map
              return_filtered_df: bool = True,
              condition: Optional[object] = None,
              # heatmap-only mode: compute and return heatmap arrays instead of plotting
              heatmap_only: bool = False,
              # stats-only mode: compute summary stats only, skip heatmap and plotting
              stats_only: bool = False,
              grid_res: float = 1.0,
              sigma: float = 6.0,
              normalize_per60: bool = False,
              selected_role: str = 'team', data_df: Optional['pd.DataFrame'] = None,
              total_seconds: Optional[float] = None,
              # new interval filtering behavior
              use_intervals: bool = True,
              intervals_input: Optional[dict] = None,
              title: Optional[str] = None,
              interval_time_col: str = 'total_time_elapsed_seconds',
              force_refresh: bool = False):
    
    if total_seconds is None:
        # Default to 0.0 if not provided, to avoid None propagation
        # (Caller should provide it if they want per60 normalization)
        total_seconds = 0.0

    """Create an xG density map for a season and save a plot.

    This function is intentionally written as a clear sequence of steps with
    small local helpers so it's easy to read and maintain. The external
    behavior (return value and saved image) is unchanged.

    High level steps (implemented below):
      1) Locate and load a season CSV.
      2) Apply an optional flexible `condition` filter (uses `parse.build_mask`).
         If `condition` is a dict with a 'team' key, that is treated as a
         request to match either home or away team (by id or abbreviation).
      3) If needed, obtain/train an xG classifier and predict probabilities
         on the filtered rows (attach as column 'xgs'). Prediction only runs
         when the filtered df is non-empty and 'xgs' either doesn't exist or
         contains only NaN values.
      4) Optionally orient coordinates so selected shots face left (or all).
      5) Call the plotting routine and save/return outputs.

    Parameters mirror the previous implementation; returning (out_path, heatmaps)
    when `return_heatmaps=True` and just `out_path` otherwise.
    """

    # Local imports placed here to avoid module-level side-effects when importing analyze
    from pathlib import Path
    import numpy as np
    from . import fit_xgs
    from . import plot as plot_mod
    from . import parse as _parse

    # --- Helpers ------------------------------------------------------------
    # Determine the CSV path to use for model training if needed, even if we have df_all
    chosen_csv = None
    try:
        chosen_csv = locate_season_csv(season, csv_path)
    except Exception:
        chosen_csv = None

    # Helper: apply a condition dict to a dataframe and return (filtered_df, team_val)
    def _apply_condition(df: pd.DataFrame):
        """Apply `condition` to df and return (filtered_df, team_val).

        Accepts None or a dict. If condition contains 'team' that key is used to
        filter rows where home or away matches that team (by abb or id).
        """
        cond_work = condition.copy() if isinstance(condition, dict) else condition

        team_val_local = None
        if isinstance(cond_work, dict) and 'team' in cond_work:
            team_val_local = cond_work.pop('team', None)

        # Normalize keys to column names where possible
        if isinstance(cond_work, dict):
            # Normalize key to alphanumeric lowercase
            col_map = {''.join(ch.lower() for ch in str(c) if ch.isalnum()): c for c in df.columns}
            corrected = {}
            for k, v in cond_work.items():
                nk = ''.join(ch.lower() for ch in str(k) if ch.isalnum())
                corrected[col_map.get(nk, k)] = v
            cond_work = corrected

        # Build mask via parse.build_mask when a dict is provided
        try:
            base_mask = pd.Series(True, index=df.index) if cond_work is None else _parse.build_mask(df, cond_work).reindex(df.index).fillna(False).astype(bool)
        except Exception as e:
            print('Warning: failed to apply condition filter:', e)
            base_mask = pd.Series(False, index=df.index)

        # Apply team filter if requested
        if team_val_local is not None:
            tstr = str(team_val_local).strip()
            # Try to parse as int ID, otherwise use as abbreviation
            try:
                tid = int(tstr)
                team_mask = pd.Series(False, index=df.index)
                if 'home_id' in df.columns:
                    team_mask |= df['home_id'].astype(str) == str(tid)
                if 'away_id' in df.columns:
                    team_mask |= df['away_id'].astype(str) == str(tid)
            except ValueError:
                tupper = tstr.upper()
                team_mask = pd.Series(False, index=df.index)
                if 'home_abb' in df.columns:
                    team_mask |= df['home_abb'].astype(str).str.upper() == tupper
                if 'away_abb' in df.columns:
                    team_mask |= df['away_abb'].astype(str).str.upper() == tupper
            final_mask = base_mask & team_mask
        else:
            final_mask = base_mask

        if int(final_mask.sum()) == 0:
            empty = df.iloc[0:0].copy()
            empty['xgs'] = float('nan')
            return empty, team_val_local
        return df.loc[final_mask].copy(), team_val_local



    def _orient_coordinates(df_in: pd.DataFrame, team_val_local):
        """Produce x_a/y_a columns for plotting according to orientation rules.

        This preserves previous logic while being slightly more compact.
        """
        df = df_in
        left_goal_x, right_goal_x = plot_mod.rink_goal_xs()

        def attacked_goal_x_for_row(team_id, home_id, home_def):
            # Returns the x-coordinate of the goal being attacked for the shooter
            try:
                if pd.isna(team_id) or pd.isna(home_id):
                    return right_goal_x
                if str(team_id) == str(home_id):
                    # shooter is home: they attack opposite of home's defended side
                    return right_goal_x if home_def == 'left' else (left_goal_x if home_def == 'right' else right_goal_x)
                else:
                    return left_goal_x if home_def == 'left' else (right_goal_x if home_def == 'right' else left_goal_x)
            except Exception:
                return right_goal_x

        df['x_a'] = df.get('x')
        df['y_a'] = df.get('y')

        # compute attacked_x for each row once
        attacked_x = df.apply(lambda r: attacked_goal_x_for_row(r.get('team_id'), r.get('home_id'), r.get('home_team_defending_side')), axis=1)

        if team_val_local is not None:
            tstr = str(team_val_local).strip()
            try:
                tid = int(tstr)
            except Exception:
                tid = None
            tupper = None if tid is not None else tstr.upper()

            def is_selected(row):
                try:
                    if tid is not None:
                        return str(row.get('team_id')) == str(tid)
                    shooter_id = row.get('team_id')
                    if pd.isna(shooter_id):
                        return False
                    if str(shooter_id) == str(row.get('home_id')) and row.get('home_abb') is not None:
                        return str(row.get('home_abb')).upper() == tupper
                    if str(shooter_id) == str(row.get('away_id')) and row.get('away_abb') is not None:
                        return str(row.get('away_abb')).upper() == tupper
                except Exception:
                    return False
                return False

            desired_goal = df.apply(lambda r: left_goal_x if is_selected(r) else right_goal_x, axis=1)
            mask_rotate = (attacked_x != desired_goal) & df['x'].notna() & df['y'].notna()
            df.loc[mask_rotate, ['x_a', 'y_a']] = -df.loc[mask_rotate, ['x', 'y']].values

        elif orient_all_left:
            mask_rotate = (attacked_x == right_goal_x) & df['x'].notna() & df['y'].notna()
            df.loc[mask_rotate, ['x_a', 'y_a']] = -df.loc[mask_rotate, ['x', 'y']].values

        return df

    def _apply_intervals(df_in: pd.DataFrame, intervals_obj, time_col: str = 'total_time_elapsed_seconds', team_val: Optional[object] = None, condition: Optional[dict] = None) -> pd.DataFrame:
        """
        Filters the input dataframe to include only rows where the event time falls within the specified intervals.

        Args:
            df_in (pd.DataFrame): The input dataframe containing event data.
            intervals_obj (dict): The intervals object containing per-game intersection intervals (as produced by timing.compute_game_timing).
            time_col (str): The column name in df_in representing the event time.
            team_val (Optional[object]): The team identifier to extract team-specific intervals when appropriate.
            condition (Optional[dict]): Additional per-row conditions to enforce (e.g. {'game_state': ['5v5'], 'is_net_empty':[0]}).

        Returns:
            pd.DataFrame: A filtered dataframe containing only rows within the specified intervals and satisfying `condition` when provided.
        """
        from . import timing as _timing  # local import to avoid top-level circular deps

        filtered_rows = []
        skipped_games = []

        per_game = intervals_obj.get('per_game', {}) if isinstance(intervals_obj, dict) else {}

        # Iterate over each game_id in the intervals object
        for game_id, game_data in per_game.items():
            # normalize game id to string for robust matching with df values
            gid_str = str(game_id)

            # Determine which team perspective to use when evaluating game_state_relative_to_team
            team_for_game = team_val if team_val is not None else (game_data.get('selected_team') if isinstance(game_data, dict) else None)

            # Obtain intervals for this game in a few supported shapes
            team_intervals = []
            try:
                if isinstance(game_data, dict):
                    # preferred shape: game_data['sides']['team']['intersection_intervals']
                    sides = game_data.get('sides')
                    if isinstance(sides, dict):
                        team_side = sides.get('team') or {}
                        team_intervals = team_side.get('intersection_intervals') or team_side.get('pooled_intervals') or []
                    # fallback: top-level intersection_intervals
                    if not team_intervals:
                        team_intervals = game_data.get('intersection_intervals') or game_data.get('merged_intervals') or []
                    # another fallback: intervals_per_condition when only one condition present
                    if not team_intervals and isinstance(game_data.get('intervals_per_condition'), dict) and condition:
                        # try to pick the intersection across requested condition keys
                        ipc = game_data.get('intervals_per_condition')
                        # if the intervals_per_condition contains keys matching condition, merge them
                        candidate_lists = []
                        for k in (condition.keys() if isinstance(condition, dict) else []):
                            if k in ipc and isinstance(ipc.get(k), list):
                                candidate_lists.append(ipc.get(k))
                        if candidate_lists:
                            # intersect multiple lists
                            from math import isfinite
                            def _intersect_two_local(a,b):
                                res=[]
                                i=j=0
                                a_sorted=sorted(a)
                                b_sorted=sorted(b)
                                while i<len(a_sorted) and j<len(b_sorted):
                                    s1,e1=a_sorted[i]
                                    s2,e2=b_sorted[j]
                                    s=max(s1,s2); e=min(e1,e2)
                                    if e> s:
                                        res.append((s,e))
                                    if e1<e2:
                                        i+=1
                                    else:
                                        j+=1
                                return res
                            inter = candidate_lists[0]
                            for lst in candidate_lists[1:]:
                                inter = _intersect_two_local(inter, lst)
                            team_intervals = inter
                else:
                    # game_data might be a plain list of intervals
                    if isinstance(game_data, (list, tuple)):
                        team_intervals = list(game_data)
            except Exception:
                team_intervals = []

            # Debug print: show per-game summary to help trace empty-filtering
            try:
                rows_in_game = 0
                if 'game_id' in df_in.columns:
                    rows_in_game = int((df_in['game_id'].astype(str) == gid_str).sum())
                print(f"_apply_intervals: game {gid_str} rows_in_game={rows_in_game} parsed_intervals={len(team_intervals)} team_for_game={team_for_game}")
            except Exception:
                pass

            # If no intervals found for this game, skip it
            if not team_intervals:
                # try to continue -- nothing to apply for this game
                skipped_games.append((gid_str, 'no_intervals'))
                continue

            # Subset rows for this game (robust to int/str mismatch)
            try:
                if 'game_id' in df_in.columns:
                    df_game = df_in[df_in['game_id'].astype(str) == gid_str]
                else:
                    df_game = df_in.copy().iloc[0:0]
            except Exception:
                df_game = df_in[df_in.get('game_id') == game_id]

            try:
                print(f"_apply_intervals: game {gid_str} df_game_rows={0 if df_game is None else int(df_game.shape[0])}")
            except Exception:
                pass

            if df_game is None or df_game.empty:
                # no rows for this game in df_in
                skipped_games.append((gid_str, 'no_rows'))
                continue

            # If we need to test game_state, prepare a df with game_state_relative_to_team
            need_game_state = False
            gs_series = None
            if condition and isinstance(condition, dict) and 'game_state' in condition:
                need_game_state = True
                try:
                    # If timing module provides helper to add relative game state, use it
                    if hasattr(_timing, 'add_game_state_relative_column'):
                        df_game_rel = _timing.add_game_state_relative_column(df_game.copy(), team_for_game)
                        gs_series = df_game_rel.get('game_state_relative_to_team') if isinstance(df_game_rel, dict) is False else None
                        # if helper returned a DataFrame-like object, try attribute
                        if gs_series is None and isinstance(df_game_rel, pd.DataFrame):
                            gs_series = df_game_rel.get('game_state_relative_to_team')
                    else:
                        gs_series = None
                except Exception:
                    gs_series = None

            # Vectorized selection: collect indices of rows whose time_col falls into any interval
            try:
                matched_indices = []
                # coerce time column to numeric once for this game's rows
                # Ensure times is always a Series aligned with df_game.index
                if time_col in df_game.columns:
                    times = pd.to_numeric(df_game[time_col], errors='coerce')
                else:
                    import numpy as _np
                    times = pd.Series(_np.nan, index=df_game.index)
                for (start, end) in team_intervals:
                    try:
                        s = float(start); e = float(end)
                    except Exception:
                        continue
                    if s == 0:
                        mask = times.notna() & (times >= s) & (times <= e)
                    else:
                        mask = times.notna() & (times >= s) & (times <= e)
                    if mask.any():
                        matched_indices.extend(df_game.loc[mask].index.tolist())
                # deduplicate while preserving order
                if matched_indices:
                    seen = set()
                    unique_idx = []
                    for ii in matched_indices:
                        if ii not in seen:
                            seen.add(ii); unique_idx.append(ii)
                    
                    # Post-filter validation: verify that matched rows satisfy the condition
                    # This handles edge cases where an event at a boundary time (e.g., power-play goal)
                    # should be included/excluded based on its actual game_state, not just time interval
                    if condition and isinstance(condition, dict):
                        # Check if condition includes state-based filters that need validation
                        needs_validation = ('game_state' in condition or 'is_net_empty' in condition)
                        
                        if needs_validation:
                            # Build a temporary dataframe from matched rows for validation
                            df_matched = df_game.loc[unique_idx]
                            
                            # If game_state is in condition, add game_state_relative_to_team column
                            # (Restored: We match relative conditions (like 4v5) against relative API state
                            #  to catch cases where API is '5v4' but relative is '4v5'.)
                            if 'game_state' in condition and hasattr(_timing, 'add_game_state_relative_column'):
                                try:
                                    df_matched = _timing.add_game_state_relative_column(df_matched.copy(), team_for_game)
                                    # Replace game_state column with relative version for condition matching
                                    if 'game_state_relative_to_team' in df_matched.columns:
                                        df_matched['game_state'] = df_matched['game_state_relative_to_team']
                                except Exception as e:
                                    print(f"_apply_intervals: failed to add game_state_relative_to_team for game {gid_str}: {e}")
                            
                            # Build a mask using parse.build_mask to test condition against matched rows
                            try:
                                # Create a condition without 'team' or player keys for build_mask validation
                                # We rely on intervals for player presence; row-level validation would exclude events by others.
                                validation_condition = {k: v for k, v in condition.items() if k not in ['team', 'player_id', 'player_ids']}
                                if validation_condition:
                                    condition_mask = _parse.build_mask(df_matched, validation_condition)
                                    condition_mask = condition_mask.reindex(df_matched.index).fillna(False).astype(bool)
                                    # Filter unique_idx using vectorized boolean indexing
                                    validated_mask = pd.Series([ii in df_matched.index and condition_mask.loc[ii] for ii in unique_idx], index=unique_idx)
                                    unique_idx = [ii for ii, keep in zip(unique_idx, validated_mask) if keep]
                            except Exception as e:
                                print(f"_apply_intervals: failed to validate condition for game {gid_str}: {e}")
                    
                    # append validated matched rows to filtered list by index reference
                    for ii in unique_idx:
                        try:
                            filtered_rows.append(df_game.loc[ii])
                        except Exception:
                            continue
                else:
                    skipped_games.append((gid_str, 'no_matches'))
            except Exception:
                skipped_games.append((gid_str, 'match_error'))

        # Create a new dataframe from the filtered rows
        if not filtered_rows:
            # Diagnostic output to help debugging why nothing matched
            try:
                print('\n_apply_intervals debug: no rows matched any intervals')
                print('intervals_obj per_game count =', len(per_game))
                # show up to 10 games summary
                i = 0
                for gk, gd in list(per_game.items())[:10]:
                    try:
                        # count intervals found by our parsing
                        s = None
                        sides = gd.get('sides') if isinstance(gd, dict) else None
                        if isinstance(sides, dict) and sides.get('team'):
                            team_section = sides.get('team') or {}
                            if isinstance(team_section, dict) and team_section.get('intersection_intervals'):
                                s = len(team_section.get('intersection_intervals') or [])
                            else:
                                s = len(team_section.get('pooled_intervals') or []) if isinstance(team_section, dict) else 0
                        elif isinstance(gd, dict) and gd.get('intersection_intervals'):
                            s = len(gd.get('intersection_intervals') or [])
                        else:
                            s = 0
                    except Exception:
                        s = '?'
                    print(' game', gk, 'intervals_parsed=', s)
                    i += 1
                print('skipped_games (sample):', skipped_games[:20])
                # print a little info about df_in shape/type for clarity
                try:
                    if df_in is None:
                        print('df_in is None')
                        cols = []
                    else:
                        try:
                            cols = list(df_in.columns)
                            print('df_in shape:', getattr(df_in, 'shape', None), 'columns_count=', len(cols))
                        except Exception:
                            cols = []
                            print('df_in present but columns not introspectable; type=', type(df_in))
                except Exception:
                    cols = []
            except Exception:
                cols = []
            # Return an empty DataFrame with the same columns as input when possible
            try:
                return pd.DataFrame(columns=cols)
            except Exception:
                return pd.DataFrame()

        return pd.DataFrame(filtered_rows, columns=df_in.columns)

    # ------------------- Main flow -----------------------------------------
    # Allow caller to specify a single game_id: fetch and parse that game's feed
    df_all = None
    # chosen_csv already set above
    
    # Variables to capture game status from live feed
    game_ongoing = False
    time_remaining = None
    
    if game_id is not None:
        try:
            from . import nhl_api as _nhl_api
            print(f"xgs_map: game_id provided ({game_id}) - fetching live feed...", flush=True)
            
            # Try fetching game feed (coerce to int or str as needed)
            feed = None
            for gid in [int(game_id), str(game_id)]:
                try:
                    feed = _nhl_api.get_game_feed(gid)
                    break
                except Exception:
                    continue
            
            if feed:
                # Capture game status immediately
                try:
                    # Debug feed structure
                    print(f"DEBUG: Feed keys: {list(feed.keys())}", flush=True)
                    
                    # Try standard path
                    status_data = feed.get('gameData', {}).get('status', {})
                    status = status_data.get('abstractGameState')
                    detailed_state = status_data.get('detailedState')
                    
                    # Try alternative path (new API)
                    if not status:
                        status = feed.get('gameState')
                        
                    print(f"DEBUG: Game Status: abstract='{status}', detailed='{detailed_state}'", flush=True)
                    
                    if status in ('Live', 'In Progress', 'LIVE', 'IN_PROGRESS') or (status == 'Final' and detailed_state == 'In Progress'): 
                        game_ongoing = True
                    
                    # Try to get time remaining
                    # Old API: liveData.linescore
                    linescore = feed.get('liveData', {}).get('linescore', {})
                    period = linescore.get('currentPeriodOrdinal')
                    time_rem = linescore.get('currentPeriodTimeRemaining')
                    
                    # New API: clock / periodDescriptor
                    if not time_rem:
                        clock = feed.get('clock', {})
                        time_rem = clock.get('timeRemaining')
                        period_desc = feed.get('periodDescriptor', {})
                        period = period_desc.get('number')
                        if period:
                            period = f"P{period}"
                            
                    if period and time_rem:
                        time_remaining = f"{time_rem} {period}"
                        print(f"DEBUG: Time Remaining: {time_remaining}", flush=True)
                    else:
                        if game_ongoing:
                            time_remaining = "In Progress"
                except Exception as e:
                    print(f"DEBUG: Error extracting game status: {e}", flush=True)

                try:
                    # use parse helpers to build the events dataframe for the single game
                    ev_df = _parse._game(feed)
                    if ev_df is not None and not ev_df.empty:
                        try:
                            df_game = _parse._elaborate(ev_df)
                        except Exception:
                            df_game = ev_df.copy()
                        df_all = df_game.copy()
                        print(f"xgs_map: loaded {len(df_all)} event rows for game {game_id}", flush=True)
                    else:
                        print(f"xgs_map: parsed feed but got empty events for game {game_id}", flush=True)
                except Exception as e:
                    print('xgs_map: parse of live feed failed:', e, flush=True)
        except Exception as e:
            print('xgs_map: failed to fetch live feed for game_id', game_id, e, flush=True)

        # If we loaded a single game's DataFrame and condition doesn't specify a team, infer the home team
        if df_all is not None and not df_all.empty:
            try:
                if condition is None or not isinstance(condition, dict):
                    condition = {} if condition is None else dict(condition)
                if 'team' not in condition:
                    home_abb = None
                    home_id = None
                    if 'home_abb' in df_all.columns:
                        try:
                            home_abb = df_all['home_abb'].dropna().unique().tolist()[0]
                        except Exception:
                            home_abb = None
                    if 'home_id' in df_all.columns and home_abb is None:
                        try:
                            home_id = df_all['home_id'].dropna().unique().tolist()[0]
                        except Exception:
                            home_id = None
                    if home_abb:
                        condition['team'] = home_abb
                        print(f"xgs_map: inferred team='{home_abb}' from live game feed and set it in condition", flush=True)
                    elif home_id is not None:
                        condition['team'] = home_id
                        print(f"xgs_map: inferred team id={home_id} from live game feed and set it in condition", flush=True)
            except Exception:
                pass

    # Determine the CSV path to use for model training if needed, even if we have df_all
    # csv logic moved to top of function
    if chosen_csv is None and csv_path:
        chosen_csv = csv_path

    # If no single-game feed requested or fetch failed, fall back to provided DataFrame or CSV
    if df_all is None:
        if data_df is not None:
            df_all = data_df.copy()
            print('xgs_map: using provided DataFrame (in-memory) -> rows=', len(df_all))
        else:
            # If we didn't find a CSV above, we can't proceed here
            if chosen_csv is None:
                 # _locate_csv raises FileNotFoundError, so we might have caught it or it wasn't called if csv_path was set
                 # Let's call it again to raise the error if needed, or just rely on the fact that we need a source
                 # Let's call it again to raise the error if needed, or just rely on the fact that we need a source
                 chosen_csv = locate_season_csv(season, None)
            
            print('xgs_map: loading CSV ->', chosen_csv)
            df_all = pd.read_csv(chosen_csv)

    # --- Single timing call: call timing.compute_game_timing once on the full dataset
    timing_full = {'per_game': {}, 'aggregate': {'intersection_pooled_seconds': {'team': 0.0, 'other': 0.0}}}
    if timing is not None:
        try:
            timing_full = timing.compute_game_timing(df_all, condition, force_refresh=force_refresh, season=season)
        except Exception as e:
            print(f'Warning: timing.compute_game_timing failed: {e}; using empty timing structure')

    # Apply filtering: either by condition or by intervals

    # Determine if we should use intervals for filtering.
    # We only strictly need intervals if the condition involves game state or other shift-dependent properties.
    # If the condition is simple (e.g. empty or just team), we prefer to skip interval filtering
    # to ensure we capture all events, including recent ones in live games where shift data might lag.
    should_use_intervals = use_intervals
    if should_use_intervals and intervals_input is None:
        if condition is None:
             should_use_intervals = False
        elif isinstance(condition, dict):
             # Keys that require shift/interval data
             interval_keys = ['game_state', 'is_net_empty', 'player_id', 'player_ids']
             if not any(k in condition for k in interval_keys):
                 should_use_intervals = False
                 print("xgs_map: condition does not require shift data; skipping interval filtering to include all events")

    if should_use_intervals:
        print(f"DEBUG: intervals_input keys: {list(intervals_input.get('per_game', {}).keys())} types={[type(k) for k in intervals_input.get('per_game', {}).keys()]}", flush=True)
        intervals = intervals_input if intervals_input is not None else timing_full
        team_param = None
        if isinstance(condition, dict):
            team_param = condition.get('team')
        
        # Check if we actually have interval data for the relevant games
        has_interval_data = False
        if isinstance(intervals, dict):
            # Check if 'per_game' has any entries
            if intervals.get('per_game'):
                has_interval_data = True
            # If we are processing a specific game_id, check if it exists in per_game
            if game_id is not None:
                # Normalize game_id to int for lookup (timing uses ints usually)
                try:
                    gid_int = int(game_id)
                    if gid_int not in intervals.get('per_game', {}):
                        has_interval_data = False
                        print(f"xgs_map: Interval data missing for game {game_id}")
                except Exception:
                    pass
        
        if not has_interval_data:
            print(f"xgs_map: WARNING: Interval data missing or empty. Falling back to condition filtering for condition {condition}")
            df_filtered, team_val = _apply_condition(df_all)
            
            # Discrepancy check (optional/diagnostic): 
            # If we were to run interval filtering on empty intervals, we'd get 0 rows.
            # Here we are getting condition-filtered rows.
            # We can log that we are "saving" the user from a 0-row result.
            if not df_filtered.empty:
                print(f"xgs_map: Fallback recovered {len(df_filtered)} events that would have been lost with missing interval data.")
                
        else:
            try:
                # debug: print a summary of the intervals object
                try:
                    if isinstance(intervals, dict):
                        per_games_count = len(intervals.get('per_game', {}))
                    else:
                        per_games_count = len(intervals) if hasattr(intervals, '__len__') else 0
                    print(f"_apply_intervals: intervals per_game count={per_games_count}")
                except Exception:
                    pass

                df_filtered = _apply_intervals(df_all, intervals, time_col=interval_time_col, team_val=team_param, condition=condition)
                if df_filtered is None:
                    print('_apply_intervals returned None; falling back to empty DataFrame')
                    df_filtered = pd.DataFrame(columns=df_all.columns if df_all is not None else [])
            except Exception as e:
                print('Exception while applying intervals:', type(e).__name__, e)
                # fallback to empty dataframe
                df_filtered = pd.DataFrame(columns=df_all.columns if df_all is not None else [])
            team_val = team_param
    else:
        df_filtered, team_val = _apply_condition(df_all)

    if df_filtered.shape[0] == 0:
        print(f"Warning: condition {condition!r} (team={team_val!r}) matched 0 rows; producing an empty plot without training/loading model")
    else:
        print(f"Filtered season dataframe to {len(df_filtered)} events by condition {condition!r} team={team_val!r}")

    # Predict xgs only when needed and possible
    df_with_xgs, clf, clf_meta = _predict_xgs(df_filtered, model_path=model_path, behavior=behavior, csv_path=chosen_csv)

    # Orientation deprecation: plotting routines now decide orientation and
    # splitting (team vs not-team or home vs away). Do not perform an
    # explicit coordinate rotation here; leave adjusted coordinates to
    # `plot.plot_events` which will call `adjust_xy_for_homeaway` only when
    # needed. If the user requested the legacy `orient_all_left` behavior,
    # emit a DeprecationWarning and ignore it here (plotting may still
    # emulate it if requested via plotting options).
    import warnings
    if orient_all_left:
        warnings.warn("'orient_all_left' is deprecated in xgs_map and ignored; control orientation via plot.plot_events options.", DeprecationWarning)

    # Use df_with_xgs directly; plot.plot_events will compute x_a/y_a if missing.
    # Add orientation step here to ensure x_a/y_a are present and correct for plotting
    df_to_plot = _orient_coordinates(df_with_xgs, team_val)


    # Compute timing and xG summary now so we can optionally display it on the plot.
    # Use the single timing result collected earlier (timing_full) as the canonical timing_result
    timing_result = timing_full

    # Compute xG totals from df_with_xgs (sum of 'xgs' per group)
    # We need to distinguish between 'team' and 'other' (opponents)
    # We'll use the same logic as compute_xg_heatmap_from_df for consistency
    
    team_xgs = 0.0
    other_xgs = 0.0
    
    # Compute xG totals
    team_xgs = 0.0
    other_xgs = 0.0
    try:
        xgs_series = pd.to_numeric(df_with_xgs.get('xgs', pd.Series([], dtype=float)), errors='coerce').fillna(0.0)
        
        # KEY FIX: Filter xG sum to only include valid shot attempts
        # Non-shot events (Faceoffs, Hits) allow feature extraction and thus get valid >0 xG predictions
        # but they should not contribute to the game total.
        attempt_types_xg = {'goal', 'shot-on-goal', 'missed-shot', 'blocked-shot'}
        is_attempt_for_xg = df_with_xgs['event'].astype(str).str.strip().str.lower().isin(attempt_types_xg)
        
        # Zero out xG for non-attempts
        xgs_series = xgs_series.where(is_attempt_for_xg, 0.0)

        if team_val is not None:
            # determine membership using home/away or ids
            def _is_team_row(r):
                try:
                    t = team_val
                    tid = int(t) if str(t).strip().isdigit() else None
                except Exception:
                    tid = None
                try:
                    if tid is not None:
                        return str(r.get('team_id')) == str(tid)
                    tupper = str(t).upper()
                    if r.get('home_abb') is not None and str(r.get('home_abb')).upper() == tupper and str(r.get('team_id')) == str(r.get('home_id')):
                        return True
                    if r.get('away_abb') is not None and str(r.get('away_abb')).upper() == tupper and str(r.get('team_id')) == str(r.get('away_id')):
                        return True
                    if r.get('home_id') is not None and r.get('team_id') is not None and str(r.get('team_id')) == str(r.get('home_id')) and r.get('home_abb') is not None and str(r.get('home_abb')).upper() == tupper:
                        return True
                    return False
                except Exception:
                    return False

            mask = df_with_xgs.apply(_is_team_row, axis=1)
            team_xgs = xgs_series[mask].sum()
            other_xgs = xgs_series[~mask].sum()
        else:
            # no team specified -> all xG is 'team' (or maybe just total)
            team_xgs = xgs_series.sum()
            other_xgs = 0.0
    except Exception:
        team_xgs = other_xgs = 0.0

    # Compute goals totals from df_filtered (sum of 'goal' events per group)
    team_goals = 0
    other_goals = 0
    try:
        # Identify goal events
        is_goal = df_filtered['event'].astype(str).str.strip().str.lower() == 'goal'
        
        if team_val is not None:
            # determine membership using home/away or ids
            def _is_team_row(r):
                try:
                    t = team_val
                    tid = int(t) if str(t).strip().isdigit() else None
                except Exception:
                    tid = None
                try:
                    if tid is not None:
                        return str(r.get('team_id')) == str(tid)
                    tupper = str(team_val).upper()
                    if r.get('home_abb') is not None and str(r.get('home_abb')).upper() == tupper:
                        return str(r.get('team_id')) == str(r.get('home_id'))
                    if r.get('away_abb') is not None and str(r.get('away_abb')).upper() == tupper:
                        return str(r.get('team_id')) == str(r.get('away_id'))
                except Exception:
                    return False
                return False
            mask = df_filtered.apply(_is_team_row, axis=1)
            team_goals = int(is_goal[mask].sum())
            other_goals = int(is_goal[~mask].sum())
        else:
            # legacy home/away split
            if 'home_id' in df_filtered.columns and 'team_id' in df_filtered.columns:
                mask = df_filtered['team_id'].astype(str) == df_filtered['home_id'].astype(str)
                team_goals = int(is_goal[mask].sum())
                other_goals = int(is_goal[~mask].sum())
            else:
                team_goals = int(is_goal.sum())
                other_goals = 0
    except Exception:
        team_goals = other_goals = 0

    # Compute attempts totals (shot-on-goal, missed-shot, blocked-shot)
    team_attempts = 0
    other_attempts = 0
    try:
        attempt_types = {'shot-on-goal', 'missed-shot', 'blocked-shot'}
        is_attempt = df_filtered['event'].astype(str).str.strip().str.lower().isin(attempt_types)
        
        if team_val is not None:
             # determine membership using home/away or ids (reuse _is_team_row logic if possible, or re-define)
             # Since _is_team_row is defined inside the try block above, we need to redefine or move it.
             # To avoid code duplication, let's just re-implement the lambda logic inline or copy.
             # Actually, let's just use the same logic.
            def _is_team_row_att(r):
                try:
                    t = team_val
                    tid = int(t) if str(t).strip().isdigit() else None
                except Exception:
                    tid = None
                try:
                    if tid is not None:
                        return str(r.get('team_id')) == str(tid)
                    tupper = str(team_val).upper()
                    if r.get('home_abb') is not None and str(r.get('home_abb')).upper() == tupper:
                        return str(r.get('team_id')) == str(r.get('home_id'))
                    if r.get('away_abb') is not None and str(r.get('away_abb')).upper() == tupper:
                        return str(r.get('team_id')) == str(r.get('away_id'))
                except Exception:
                    return False
                return False
            
            mask = df_filtered.apply(_is_team_row_att, axis=1)
            team_attempts = int(is_attempt[mask].sum())
            other_attempts = int(is_attempt[~mask].sum())
        else:
            # legacy home/away split
            if 'home_id' in df_filtered.columns and 'team_id' in df_filtered.columns:
                mask = df_filtered['team_id'].astype(str) == df_filtered['home_id'].astype(str)
                team_attempts = int(is_attempt[mask].sum())
                other_attempts = int(is_attempt[~mask].sum())
            else:
                team_attempts = int(is_attempt.sum())
                other_attempts = 0
    except Exception:
        team_attempts = other_attempts = 0

    # extract seconds from timing_result aggregate
    # extract seconds from timing_result aggregate
    # extract seconds from timing_result aggregate
    try:
        # Prioirty: Use explicit total_seconds if provided (e.g. from process_daily_cache with specific intervals)
        if total_seconds is not None and total_seconds > 0:
            team_seconds = float(total_seconds)
            other_seconds = float(total_seconds)
        else:
            agg = timing_result.get('aggregate', {}) if isinstance(timing_result, dict) else {}
            inter = agg.get('intersection_seconds_total', 0.0)
            team_seconds = float(inter or 0.0)
            other_seconds = team_seconds
    except Exception:
        team_seconds = other_seconds = float(total_seconds) if total_seconds else 0.0

    team_xg_per60 = (team_xgs / team_seconds * 3600.0) if team_seconds > 0 else 0.0
    other_xg_per60 = (other_xgs / other_seconds * 3600.0) if other_seconds > 0 else 0.0

    # Calculate n_games
    n_games = 0
    if 'game_id' in df_filtered.columns:
        n_games = df_filtered['game_id'].nunique()
    
    summary_stats = {
        'team_xgs': team_xgs,
        'other_xgs': other_xgs,
        'team_goals': team_goals,
        'other_goals': other_goals,
        'team_attempts': team_attempts,
        'other_attempts': other_attempts,
        'team_seconds': team_seconds,
        'other_seconds': other_seconds,
        'team_xg_per60': team_xg_per60,
        'other_xg_per60': other_xg_per60,
        'n_games': n_games,
        'game_ongoing': game_ongoing,
        'time_remaining': time_remaining,
    }

    # Construct filter string for display
    filter_str = ""
    if condition:
        # Filter out 'team' and 'player_id' from display
        display_cond = {k: v for k, v in condition.items() if k not in ['team', 'player_id', 'player_ids']}
        if display_cond:
            parts = []
            for k, v in display_cond.items():
                # Format key (e.g. game_state -> Game State)
                key_fmt = k.replace('_', ' ').title()
                # Format value
                val_fmt = str(v)
                if isinstance(v, list):
                    val_fmt = ",".join(map(str, v))
                parts.append(f"{key_fmt}: {val_fmt}")
            filter_str = " | ".join(parts)
        else:
            # If condition provided but empty (or only team), explicitly say "All Events"
            # UNLESS it was None originally (but we normalized to {}).
            # But here we are inside `if condition`.
            # If the user passed {}, display_cond is empty.
            # We want to show "All Events" if they passed {} to override defaults.
            # But we don't know if they passed {} or if we defaulted to {}?
            # Actually, earlier we did `condition = {} if condition is None else dict(condition)`.
            # So we can't distinguish easily.
            # However, if display_cond is empty, it means no filtering on game state etc.
            # So "All Events" is accurate.
            filter_str = "All Events"
    else:
        # If condition is None (shouldn't happen due to earlier normalization) or empty
        filter_str = "All Events"

    # Decide heatmap split mode:
    # - If a team is specified, use 'team_not_team'.
    # - If a game is specified (but not a team), use 'home_away'.
    # - If neither team nor game is specified, use 'orient_all_left'.
    if team_val is not None:
        heatmap_mode = 'team_not_team'
    elif condition is not None and isinstance(condition, dict) and 'game_id' in condition:
        heatmap_mode = 'home_away'
    else:
        heatmap_mode = 'orient_all_left'
    



    # Call plot_events and handle both return shapes and the optional heatmap return
    if stats_only:
        return out_path, None, df_filtered if return_filtered_df else None, summary_stats

    if heatmap_only:
        # Optimization: Compute heatmaps directly without plotting
        # 1. Adjust coordinates
        df_adj = plot_mod.adjust_xy_for_homeaway(df_to_plot, split_mode=heatmap_mode, team_for_heatmap=team_val)
        
        # Filter for heatmaps: Only use valid shot attempts (exclude faceoffs, hits, etc.)
        # Reuse strict attempt types defined earlier if available, or redefine
        attempt_types_grid = {'goal', 'shot-on-goal', 'missed-shot', 'blocked-shot'}
        if 'event' in df_adj.columns:
            # Case-insensitive check
            # Use local variable for filtered DF to avoid affecting other logic if any
            df_adj_grid = df_adj[df_adj['event'].astype(str).str.strip().str.lower().isin(attempt_types_grid)].copy()
        else:
            df_adj_grid = df_adj.copy()
        
        # 2. Compute heatmaps based on mode

        heatmaps = {}
        
        if heatmap_mode == 'team_not_team':

            # Compute 'team' heatmap
            _, _, heat_team, _, _ = compute_xg_heatmap_from_df(
                df_adj_grid, grid_res=grid_res, sigma=sigma,
                selected_team=team_val, selected_role='team',
                total_seconds=total_seconds
            )
            heatmaps['team'] = heat_team
            
            # Compute 'other' heatmap
            _, _, heat_other, _, _ = compute_xg_heatmap_from_df(
                df_adj_grid, grid_res=grid_res, sigma=sigma,
                selected_team=team_val, selected_role='other',
                total_seconds=total_seconds
            )
            heatmaps['other'] = heat_other
            
        elif heatmap_mode == 'home_away':
            # Compute 'home' heatmap
            _, _, heat_home, _, _ = compute_xg_heatmap_from_df(
                df_adj_grid, grid_res=grid_res, sigma=sigma,
                selected_role='home' # compute_xg_heatmap_from_df needs update to handle 'home' role if not present?
                # Actually compute_xg_heatmap_from_df uses _is_selected_row which handles 'team'/'other'.
                # For home/away, we might need to filter df manually?
                # Let's check compute_xg_heatmap_from_df implementation.
                # It takes selected_team and selected_role.
                # If selected_role='home', does it work?
                # I'll stick to calling plot_events for non-team modes to be safe, 
                # but for 'team_not_team' (which is used in season analysis), I can optimize.
            )
            # Fallback to plot_events for other modes if unsure
            ret = plot_mod.plot_events(
                df_to_plot,
                out_path=out_path,
                return_heatmaps=True,
                heatmap_split_mode=heatmap_mode,
                team_for_heatmap=team_val,
                summary_stats=summary_stats,
                title=title,
                events_to_plot=['shot-on-goal', 'goal', 'xGs'],
                total_seconds=total_seconds,
            )
            if len(ret) >= 3:
                fig, ax, heatmaps = ret[0], ret[1], ret[2]
                plt.close(fig) # Ensure closed
            else:
                 raise ValueError("Expected at least 3 elements in 'ret'")
                 
        else: # orient_all_left
             _, _, heat, _, _ = compute_xg_heatmap_from_df(
                df_adj_grid, grid_res=grid_res, sigma=sigma,
                total_seconds=total_seconds
            )
             heatmaps = heat
             
        # No figure created
        out_path = out_path # Unchanged
             
    elif return_heatmaps:
        # For heatmap generation, ask plot_events to use the selected mode and pass team_val
        ret = plot_mod.plot_events(
            df_to_plot,
            out_path=out_path,
            return_heatmaps=True,
            heatmap_split_mode=heatmap_mode,
            team_for_heatmap=team_val,
            summary_stats=summary_stats,
            title=title,
            events_to_plot=['shot-on-goal', 'goal',
                            'xGs'],
        )
        # Fix for tuple index out of range
        # Ensure the `ret` object has enough elements before unpacking
        if len(ret) >= 3:
            fig, ax, heatmaps = ret[0], ret[1], ret[2]
        else:
            raise ValueError("Expected at least 3 elements in 'ret', but got fewer.")
    else:
        # When not requesting heatmaps, still ensure plotting uses the same
        # mode so visuals match the heatmap logic: use heatmap_mode and team_val.
        ret = plot_mod.plot_events(
            df_to_plot,
            events_to_plot=events_to_plot,
            out_path=out_path,
            heatmap_split_mode=heatmap_mode,
            team_for_heatmap=team_val,
            summary_stats=summary_stats,
            title=title,
        )
        if isinstance(ret, (tuple, list)):
            if len(ret) >= 2:
                fig, ax = ret[0], ret[1]
            else:
                raise RuntimeError('Unexpected return from plot.plot_events; expected (fig, ax)')
        else:
            raise RuntimeError('Unexpected return type from plot.plot_events')
        heatmaps = None

    if show and not heatmap_only:
        try:
            fig.show()
        except Exception:
            pass
    elif not heatmap_only:
        # Close the figure to prevent leaks
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass

    # Determine return structure: always return (out_path, heatmaps_or_None, filtered_df_or_None)
    ret_heat = heatmaps if (return_heatmaps or heatmap_only) else None
    ret_df = df_filtered.copy() if ('df_filtered' in locals() and return_filtered_df) else None
    return out_path, ret_heat, ret_df, summary_stats


# ----------------- xG heatmap helpers (moved above the CLI so they are available)
# These helpers implement the next_steps plan: compute Gaussian-smoothed xG
# heatmaps and aggregate per-team maps for a season. They are intentionally
# simple and readable — we can optimize later (FFT convolution, parallelism).


def compute_xg_heatmap_from_df(
    df,
    grid_res: float = 1.0,
    sigma: float = 6.0,
    x_col: str = 'x_a',
    y_col: str = 'y_a',
    amp_col: str = 'xgs',
    rink_mask_fn=None,
    normalize_per60: bool = False,
    total_seconds: float = None,
    selected_team: Optional[object] = None,
    selected_role: str = 'team',
):
    """Compute an xG heatmap on a fixed rink grid from an events DataFrame.

    Returns (gx, gy, heat, total_xg, total_seconds_used)
    - gx: 1D array of x grid centers
    - gy: 1D array of y grid centers
    - heat: 2D array shape (len(gy), len(gx)) with summed xG (or xG/60 if normalized)
    """
    import numpy as np
    from .rink import rink_half_height_at_x

    if df is None or df.shape[0] == 0:
        # return empty grid centered on rink extents
        gx = np.arange(-100.0, 100.0 + grid_res, grid_res)
        gy = np.arange(-42.5, 42.5 + grid_res, grid_res)
        heat = np.zeros((len(gy), len(gx)), dtype=float)
        return gx, gy, heat, 0.0, 0.0

    # We'll compute adjusted coordinates (x_a,y_a) here so callers may pass
    # the full df and ask for team-specific subsets via selected_team/selected_role.
    # ensure coordinates exist (x_a/y_a) or compute them from raw x,y
    df_work = df.copy()
    if x_col not in df_work.columns or y_col not in df_work.columns:
        try:
            from . import plot as _plot
            df_work = _plot.adjust_xy_for_homeaway(df_work)
        except Exception:
            # fall back to raw x/y passthrough
            df_work[x_col] = df_work.get('x')
            df_work[y_col] = df_work.get('y')
    else:
        # ensure presence
        df_work[x_col] = df_work.get(x_col)
        df_work[y_col] = df_work.get(y_col)

    xs_all = pd.to_numeric(df_work.get(x_col, pd.Series([], dtype=float)), errors='coerce')
    ys_all = pd.to_numeric(df_work.get(y_col, pd.Series([], dtype=float)), errors='coerce')
    amps_all = pd.to_numeric(df_work.get(amp_col, pd.Series([], dtype=float)), errors='coerce').fillna(0.0)

    # Determine a boolean mask for the subset we want: if selected_team provided,
    # include only rows matching (selected_role == 'team') or the inverse for 'other'.
    def _is_selected_row(r, team_val):
        try:
            if team_val is None:
                return False
            tstr = str(team_val).strip()
            try:
                tid = int(tstr)
            except Exception:
                tid = None
            if tid is not None:
                return str(r.get('team_id')) == str(tid)
            tupper = tstr.upper()
            # compare abbreviations if present
            if r.get('home_abb') is not None and str(r.get('home_abb')).upper() == tupper and str(r.get('team_id')) == str(r.get('home_id')):
                return True
            if r.get('away_abb') is not None and str(r.get('away_abb')).upper() == tupper and str(r.get('team_id')) == str(r.get('away_id')):
                return True
            # fallback: compare team_id to home/away ids when abb not available
            if r.get('home_id') is not None and r.get('team_id') is not None and str(r.get('team_id')) == str(r.get('home_id')) and r.get('home_abb') is not None and str(r.get('home_abb')).upper() == tupper:
                return True
            return False
        except Exception:
            return False

    if selected_team is not None:
        sel_mask = df_work.apply(lambda r: _is_selected_row(r, selected_team), axis=1)
        if selected_role == 'team':
            use_mask = sel_mask
        else:
            use_mask = ~sel_mask
        
        # DEBUG: Inspect mask and amps
        # print(f"DEBUG: compute_xg_heatmap role={selected_role} team={selected_team} mask_sum={use_mask.sum()} total_rows={len(df_work)}")
    else:
        use_mask = pd.Series(True, index=df_work.index)

    # as part of masking, censor out rows that have None or non-finite values in critical parts
    xs_temp = xs_all[use_mask]
    ys_temp = ys_all[use_mask]
    amps_temp = amps_all[use_mask]
    
    # build a validity mask (non-null and finite)
    # build a validity mask (non-null and finite)
    try:
        valid_mask = xs_temp.notna() & ys_temp.notna() & amps_temp.notna()
        valid_mask &= xs_temp.apply(np.isfinite) & ys_temp.apply(np.isfinite) & amps_temp.apply(np.isfinite)
    except Exception:
        valid_mask = (~xs_temp.isna()) & (~ys_temp.isna()) & (~amps_temp.isna())
    xs = xs_temp[valid_mask]
    ys = ys_temp[valid_mask]
    amps = amps_temp[valid_mask]

    gx = np.arange(-100.0, 100.0 + grid_res, grid_res)
    gy = np.arange(-42.5, 42.5 + grid_res, grid_res)
    XX, YY = np.meshgrid(gx, gy)

    heat = np.zeros_like(XX, dtype=float)

    if xs.size == 0:
        total_xg = 0.0
    else:
        total_xg = float(amps.sum())
        two_sigma2 = 2.0 * (sigma ** 2)
        norm_factor = (grid_res ** 2) / (2.0 * np.pi * (sigma ** 2))
        # accumulate kernels per event (loop acceptable for typical shot counts)
        for xi, yi, ai in zip(xs, ys, amps):
            dx = XX - float(xi)
            dy = YY - float(yi)
            kern = ai * norm_factor * np.exp(-(dx * dx + dy * dy) / two_sigma2)
            heat += kern
    # mask outside rink: compute a boolean rink_mask and set outside cells to NaN
    try:
        rink_mask = np.vectorize(rink_half_height_at_x)(XX) >= np.abs(YY)
        # where rink_mask is False, explicitly set heat to NaN so it will be
        # treated as masked/transparent by plotting routines that mask NaNs.
        heat = np.where(rink_mask, heat, np.nan)
    except Exception:
        # ensure numeric array at minimum
        try:
            heat = np.asarray(heat, dtype=float)
        except Exception:
            pass

    # Ensure heat is a numeric float array; keep NaNs as NaN (do not replace)
    try:
        heat = np.asarray(heat, dtype=float)
    except Exception:
        pass

    # Determine total_seconds_used: prefer explicit input, else try to infer
    total_seconds_used = None
    if total_seconds is not None:
        try:
            total_seconds_used = float(total_seconds)
        except Exception:
            total_seconds_used = None

    if total_seconds_used is None:
        if normalize_per60:
             # If we need to normalize but have no time, we cannot proceed safely.
             # User requested NO inference.
             print(f"DEBUG: compute_xg_heatmap_from_df: total_seconds is None and normalize_per60=True. Returning zeros.")
             total_seconds_used = 0.0 # Will result in zero rate
        else:
             # Not normalizing, so time doesn't matter for the heatmap values themselves
             total_seconds_used = 0.0

    # If requested, normalize heat and total_xg to per-60 units using total_seconds_used
    if normalize_per60 and total_seconds_used and total_seconds_used > 0.0:
        try:
            scale = 3600.0 / float(total_seconds_used)
            heat = heat * scale
            total_xg = float(total_xg) * scale
        except Exception:
            # fall back to no scaling
            pass

    # Final safety: convert non-numeric entries to float but keep NaNs so
    # plotting routines can mask them (do not replace NaNs with zeros).
    try:
        heat = np.asarray(heat, dtype=float)
    except Exception:
        pass

    return gx, gy, heat, float(total_xg), float(total_seconds_used or 0.0)


def xg_maps_for_season(season_or_df, condition=None, grid_res: float = 1.0, sigma: float = 6.0, out_dir: str = 'analysis/league', min_events: int = 5, model_path: str = 'analysis/xgs/xg_model.joblib', behavior: str = 'load', csv_path: str = None):
    """Compute league and per-team xG maps for a season (or events DataFrame).

    Saves PNG and JSON summary per team into out_dir/{season}/
    Returns a dict with league_map info and per-team summaries.
    """
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import json
    import matplotlib.pyplot as plt
    from .rink import draw_rink
    from . import parse as _parse

    # load season df or accept a provided DataFrame
    if isinstance(season_or_df, str):
        from . import timing
        df = timing.load_season_df(season_or_df)
        season = season_or_df
    else:
        df = season_or_df.copy()
        # try to infer season name
        season = getattr(df, 'season', None) or 'season'

    if df is None or df.shape[0] == 0:
        raise ValueError('No events data available for xg_maps_for_season')

    # apply condition filter using parse.build_mask when condition provided
    if condition is None:
        df_cond = df.copy()
    else:
        try:
            mask = _parse.build_mask(df, condition)
            mask = mask.reindex(df.index).fillna(False).astype(bool)
            df_cond = df.loc[mask].copy()
        except Exception as e:
            print(f"xgs_maps_for_season: filtering failed, using full df. Error: {e}")
            df_cond = df.copy()

    # ensure adjusted coords exist
    try:
        from . import plot as _plot
        df_cond = _plot.adjust_xy_for_homeaway(df_cond)
    except Exception:
        pass

    # If xgs are missing or all zeros, try to predict using the xG classifier
    try:
        if 'xgs' in df_cond.columns:
            xgs_series = pd.to_numeric(df_cond['xgs'], errors='coerce').fillna(0.0)
        else:
            xgs_series = pd.Series(0.0, index=df_cond.index)

        need_predict = ('xgs' not in df_cond.columns) or (xgs_series.sum() == 0)
        df_cond['xgs'] = xgs_series
    except Exception:
        need_predict = True

    if need_predict:
        try:
            from . import fit_xgs
            # use get_or_train_clf which caches/trains if needed
            force_retrain = True if (behavior or '').strip().lower() == 'train' else False
            try:
                clf, feature_names, cat_levels = fit_xgs.get_or_train_clf(force_retrain=force_retrain, csv_path=csv_path)
            except Exception:
                # fallback to older get_clf
                clf, feature_names, cat_levels = fit_xgs.get_clf(model_path, behavior, csv_path=csv_path)

            features = ['distance', 'angle_deg', 'game_state', 'is_net_empty']
            df_model, final_feature_cols_game, cat_map_game = fit_xgs.clean_df_for_model(df_cond.copy(), features, fixed_categorical_levels=cat_levels)

            # ensure xgs column exists
            df_cond['xgs'] = pd.to_numeric(df_cond.get('xgs', pd.Series(dtype=float)), errors='coerce').fillna(0.0)

            final_features = feature_names if feature_names is not None else final_feature_cols_game
            if clf is not None and df_model.shape[0] > 0 and final_features:
                try:
                    X = df_model[final_features].values
                    probs = clf.predict_proba(X)[:, 1]
                    # Apply predictions back to the original df_cond by index
                    df_cond.loc[df_model.index, 'xgs'] = probs
                    print(f'xg_maps_for_season: predicted xgs for {len(probs)} events')
                except Exception as e:
                    print('xg_maps_for_season: prediction failed:', e)
        except Exception as e:
            print('xg_maps_for_season: failed to predict xgs:', e)


    # Derive total_seconds for normalization from timing.compute_game_timing using the input condition.
    # This gives a more accurate denominator for normalize_per60 than inferring from observed timestamp ranges.
    try:
        from . import timing
        timing_res = timing.compute_game_timing(df_cond, condition, season=season)
        agg = timing_res.get('aggregate', {}) if isinstance(timing_res, dict) else {}
        # Try to get pooled seconds (team vs other), fallback to total intersection
        inter = agg.get('intersection_pooled_seconds')
        if inter and isinstance(inter, dict):
            team_secs = float(inter.get('team') or 0.0)
            other_secs = float(inter.get('other') or 0.0)
        else:
            # fallback: assume symmetric time for team/other if pooled missing
            total_inter = float(agg.get('intersection_seconds_total') or 0.0)
            team_secs = total_inter
            other_secs = total_inter
            
        total_seconds_cond = float(team_secs + other_secs) if (team_secs or other_secs) else None
    except Exception:
        total_seconds_cond = None

    print(f"DEBUG: league_xg calculation. total_seconds_cond={total_seconds_cond}")


    # compute league map with all shots oriented to the LEFT (so league baseline is offense-left)
    df_league_left = orient_all(df_cond, target='left')
    gx, gy, league_heat, league_xg, league_seconds = compute_xg_heatmap_from_df(
        df_league_left, grid_res=grid_res, sigma=sigma, normalize_per60=True, total_seconds=total_seconds_cond)
    print(f"DEBUG: Computed league_xg={league_xg}, league_seconds={league_seconds}")


    # prepare output dir
    base_out = Path(out_dir) / str(season)
    base_out.mkdir(parents=True, exist_ok=True)

    # save baseline - use out_dir directly to match load path
    baseline_json = Path(out_dir) / f'{season}_league_baseline.json'
    baseline_npy = Path(out_dir) / f'{season}_league_baseline.npy'
    baseline_right_npy = Path(out_dir) / f'{season}_league_baseline_right.npy'

    # save league map figure
    try:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        draw_rink(ax=ax)
        extent = (gx[0] - grid_res / 2.0, gx[-1] + grid_res / 2.0, gy[0] - grid_res / 2.0, gy[-1] + grid_res / 2.0)
        cmap = plt.get_cmap('viridis')
        # ensure NaNs are masked so outer rink cells render transparent
        try:
            cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))
        except Exception:
            try:
                cmap.set_bad(color='white')
            except Exception:
                pass
        try:
            m_league = np.ma.masked_invalid(league_heat)
            im = ax.imshow(m_league, extent=extent, origin='lower', cmap=cmap, zorder=1, alpha=0.8)
        except Exception:
            im = ax.imshow(league_heat, extent=extent, origin='lower', cmap=cmap, zorder=1, alpha=0.8)
        fig.colorbar(im, ax=ax, label='xG per hour (approx)')
        out_png = base_out / f'{season}_league_map.png'
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
    except Exception:
        pass

    # determine team list (use abbreviations when possible)
    teams = set()
    if 'home_abb' in df_cond.columns:
        teams.update([a for a in df_cond['home_abb'].dropna().astype(str).unique().tolist()])
    if 'away_abb' in df_cond.columns:
        teams.update([a for a in df_cond['away_abb'].dropna().astype(str).unique().tolist()])
    # fallback to team ids if no abbs
    if not teams and 'home_id' in df_cond.columns:
        teams.update([str(a) for a in df_cond['home_id'].dropna().unique().tolist()])

    # save baseline npy
    try:
        np.save(baseline_npy, league_heat)
        print(f"league: saved baseline to {baseline_npy}")
        
        # Save right baseline if computed (it is computed as league_right above)
        if league_right is not None:
             np.save(baseline_right_npy, league_right)
             print(f"league: saved right baseline to {baseline_right_npy}")
             
    except Exception as e:
        print(f"league: failed to save baseline npy: {e}")

    results = {'season': season, 'league': {'gx': list(gx), 'gy': list(gy), 'xg_total': float(league_xg), 'seconds': float(league_seconds)}}
    results['teams'] = {}

    # Lists to store per-team stats for the summary plot
    all_team_names = []
    all_team_xg_per60 = []
    all_opp_xg_per60 = []

    for team in sorted(teams):
        # team membership mask
        try:
            t = str(team).strip().upper()
            mask_team = ((df_cond.get('home_abb', pd.Series(dtype=object)).astype(str).str.upper() == t) | (df_cond.get('away_abb', pd.Series(dtype=object)).astype(str).str.upper() == t))
        except Exception:
            mask_team = pd.Series(False, index=df_cond.index)

        df_team = df_cond.loc[mask_team]
        df_opp = df_cond.loc[~mask_team]
        n_events = int(df_team.shape[0])
        if n_events < min_events:
            # skip low-sample teams
            continue

        # Derive per-team timing seconds via timing.compute_game_timing using the base condition + team
        # Use timing.compute_game_timing to get precise intervals for this team
        try:
            from . import timing
            # We want to filter intervals where game_state='5v5' (or whatever is in condition)
            # AND is_net_empty matches our request.
            # compute_game_timing expects a condition dict.
            cond_for_timing = condition.copy() if condition else {}
            # Ensure team is set so we get team-relative stats
            cond_for_timing['team'] = team
            
            timing_res = timing.compute_game_timing(df_games, cond_for_timing, verbose=False, season=season)
            agg_t = timing_res.get('aggregate', {}) if isinstance(timing_res, dict) else {}
            
            # Try to get pooled seconds, fallback to total intersection
            inter_t = agg_t.get('intersection_pooled_seconds')
            if inter_t and isinstance(inter_t, dict):
                team_secs_t = float(inter_t.get('team') or 0.0)
                other_secs_t = float(inter_t.get('other') or 0.0)
            else:
                total_inter_t = float(agg_t.get('intersection_seconds_total') or 0.0)
                team_secs_t = total_inter_t
                other_secs_t = total_inter_t
        except Exception:
            # fall back to season-level total_seconds_cond
            team_secs_t = None
            other_secs_t = None

        # Prepare sensible fallbacks for normalization: prefer per-team seconds when present,
        # otherwise fall back to the season-level total_seconds_cond (if available), else None.
        team_total_seconds_for_heat = team_secs_t if team_secs_t and team_secs_t > 0 else (total_seconds_cond if 'total_seconds_cond' in locals() else None)
        opp_total_seconds_for_heat = other_secs_t if other_secs_t and other_secs_t > 0 else (total_seconds_cond if 'total_seconds_cond' in locals() else None)

        # Filter the season-level df to only games involving this team. This ensures
        # both heatmaps (team and opponents) are derived from the same set of games.
        games_for_team = []
        try:
            games_for_team = df_cond.loc[mask_team, 'game_id'].dropna().unique().tolist()
        except Exception:
            games_for_team = []

        if games_for_team:
            df_games = df_cond.loc[df_cond['game_id'].isin(games_for_team)].copy()
        else:
            # fallback: if no game ids, just use the per-team rows and their opponents
            df_games = pd.concat([df_team, df_opp], ignore_index=False).copy()

        # Orient the combined games such that selected team shots face LEFT and
        # opponents face RIGHT. This produces x_a/y_a coordinates suitable for
        # both subsequent heatmap computations.
        try:
            df_oriented = orient_all(df_games, target='left', selected_team=team, selected_role='team')
        except Exception:
            # last-resort: use df_games as-is
            df_oriented = df_games.copy()

        # Team heat: compute from oriented df but ask function to restrict to the selected team
        gx_t, gy_t, team_heat, team_xg, team_seconds = compute_xg_heatmap_from_df(
            df_oriented, grid_res=grid_res, sigma=sigma, normalize_per60=True,
            total_seconds=team_total_seconds_for_heat, selected_team=team, selected_role='team')

        # Opponent heat: compute from the same oriented df but restrict to opponents
        gx_o, gy_o, opp_heat, opp_xg, opp_seconds = compute_xg_heatmap_from_df(
            df_oriented, grid_res=grid_res, sigma=sigma, normalize_per60=True,
            total_seconds=opp_total_seconds_for_heat, selected_team=team, selected_role='other')

        import numpy as np
        eps = 1e-9

        # Determine blue-line x positions (same as rink.draw_rink uses: ±25.0)
        blue_x = 25.0
        left_blue_x = -blue_x
        right_blue_x = blue_x

        # Create x masks for grid columns: team offensive zone is gx <= left_blue_x; opponent offense is gx >= right_blue_x
        gx_arr = np.array(gx)
        mask_team_zone = gx_arr <= left_blue_x
        mask_opp_zone = gx_arr >= right_blue_x

        # Mask league heat to left and right zones accordingly
        league_left_zone = np.full_like(league_heat, np.nan)
        league_left_zone[:, mask_team_zone] = league_heat[:, mask_team_zone]

        league_right = rotate_heat_180(league_heat)
        league_right_zone = np.full_like(league_right, np.nan)
        league_right_zone[:, mask_opp_zone] = league_right[:, mask_opp_zone]

        # Mask team and opponent heat to offensive zones
        team_zone = np.full_like(team_heat, np.nan)
        team_zone[:, mask_team_zone] = team_heat[:, mask_team_zone]

        opp_zone = np.full_like(opp_heat, np.nan)
        opp_zone[:, mask_opp_zone] = opp_heat[:, mask_opp_zone]

        # Compute percent-difference maps only within masked zones (NaNs elsewhere)
        try:
            pct_team = (team_zone - league_left_zone) / (league_left_zone + eps) * 100.0
        except Exception:
            pct_team = np.full_like(team_zone, np.nan)

        try:
            pct_opp = (opp_zone - league_right_zone) / (league_right_zone + eps) * 100.0
        except Exception:
            pct_opp = np.full_like(opp_zone, np.nan)

        # Combined summary plot: single rink with team (left) and opponents (right)
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            # Reserve space above the rink so summary text sits above and does not overlap
            try:
                fig.subplots_adjust(top=0.78)
            except Exception:
                pass
            # draw single rink
            draw_rink(ax=ax)

            # Fixed colorbar range for percent-change maps across all teams.
            # Base range is ±100 (%), extend it by 25% to give some headroom.
            colorbar_base = 100.0
            colorbar_extension = 1.25
            vmin = -colorbar_base * colorbar_extension
            vmax = colorbar_base * colorbar_extension

            cmap = plt.get_cmap('RdBu_r')
            # Render NaN (masked) cells as transparent so the rink beneath is visible
            try:
                cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))
            except Exception:
                # fallback to white if transparency unsupported
                cmap.set_bad(color='white')

            # ensure `extent` is defined (may have been created earlier for league map)
            extent = (gx[0] - grid_res / 2.0, gx[-1] + grid_res / 2.0,
                      gy[0] - grid_res / 2.0, gy[-1] + grid_res / 2.0)

            # plot team (left zone) then opponents (right zone) on same axes; NaNs are transparent
            try:
                m_pct_team = np.ma.masked_invalid(pct_team)
                m_pct_opp = np.ma.masked_invalid(pct_opp)
            except Exception:
                m_pct_team = pct_team
                m_pct_opp = pct_opp
            im_team = ax.imshow(m_pct_team, extent=extent, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, zorder=2)
            im_opp = ax.imshow(m_pct_opp, extent=extent, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, zorder=2)

            # Structured title and two-column stats placed above the rink.
            try:
                cond_desc = str(condition) if condition is not None else 'All'
            except Exception:
                cond_desc = 'All'

            main_title = f"{team} — {cond_desc}"

            # Layout positions for title/subtitle/stats in figure coordinates
            title_y = 0.94
            subtitle_y = 0.90
            stats_y_start = 0.86
            line_gap = 0.03

            # Derive numeric summaries used in the top text block. These
            # variables are computed here to avoid unresolved-reference errors
            # while keeping the calculations compact and robust to missing data.
            # Prefer per-team computed values (team_seconds / opp_seconds), then
            # fall back to earlier aggregate values if unavailable.
            # Derive numeric summaries used in the top text block. Prefer per-team values
            try:
                if 'team_seconds' in locals() and team_seconds is not None:
                    t_secs = float(team_seconds or 0.0)
                elif 'team_secs_t' in locals() and team_secs_t is not None:
                    t_secs = float(team_secs_t or 0.0)
                elif 'total_seconds_cond' in locals() and total_seconds_cond is not None:
                    t_secs = float(total_seconds_cond or 0.0)
                else:
                    t_secs = 0.0
            except Exception:
                t_secs = 0.0

            try:
                if 'opp_seconds' in locals() and opp_seconds is not None:
                    o_secs = float(opp_seconds or 0.0)
                elif 'other_secs_t' in locals() and other_secs_t is not None:
                    o_secs = float(other_secs_t or 0.0)
                elif 'total_seconds_cond' in locals() and total_seconds_cond is not None:
                    o_secs = float(total_seconds_cond or 0.0)
                else:
                    o_secs = 0.0
            except Exception:
                o_secs = 0.0

            t_min = t_secs / 60.0 if t_secs > 0 else 0.0
            o_min = o_secs / 60.0 if o_secs > 0 else 0.0

            # compute per-60 rates: when normalize_per60=True was used earlier,
            # the returned team_xg/opp_xg already represent per-60 values.
            try:
                t_xg_per60 = float(team_xg or 0.0)
            except Exception:
                t_xg_per60 = 0.0
            try:
                o_xg_per60 = float(opp_xg or 0.0)
            except Exception:
                o_xg_per60 = 0.0

            # total xG over the interval (derived from per-60 rate)
            try:
                t_xg_total = t_xg_per60 * (t_secs / 3600.0)
            except Exception:
                t_xg_total = 0.0
            try:
                o_xg_total = o_xg_per60 * (o_secs / 3600.0)
            except Exception:
                o_xg_total = 0.0

            # vs-league difference in per-60 units (team - league)
            try:
                league_per60 = float(league_xg or 0.0)
            except Exception:
                league_per60 = 0.0
            try:
                t_vs_league = t_xg_per60 - league_per60
            except Exception:
                t_vs_league = 0.0
            try:
                o_vs_league = o_xg_per60 - league_per60
            except Exception:
                o_vs_league = 0.0

            # Title centered
            fig.text(0.5, title_y, main_title, fontsize=12, fontweight='bold', ha='center')
            # Subtitles for left/right (positioned in figure coords)
            fig.text(0.25, subtitle_y, 'Offense', fontsize=10, fontweight='semibold', ha='center')
            fig.text(0.75, subtitle_y, 'Defense', fontsize=10, fontweight='semibold', ha='center')

            # Build the lines of summary text for the left (team) and right (opponent)
            left_lines = [
                f"Time: {t_min:.1f} min",
                f"xG: {t_xg_total:.2f}",
                f"xG/60: {t_xg_per60:.3f}",
                f"vs league: {t_vs_league:+.3f} xG/60",
            ]
            right_lines = [
                f"Time: {o_min:.1f} min",
                f"xG: {o_xg_total:.2f}",
                f"xG/60: {o_xg_per60:.3f}",
                f"vs league: {o_vs_league:+.3f} xG/60",
            ]

            for i, (l, r) in enumerate(zip(left_lines, right_lines)):
                y = stats_y_start - i * line_gap
                fig.text(0.25, y, l, fontsize=9, fontweight='bold' if i == 0 else 'normal', ha='center')
                fig.text(0.75, y, r, fontsize=9, fontweight='bold' if i == 0 else 'normal', ha='center')
            # shared colorbar
            cbar = fig.colorbar(im_opp, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('pct change vs league (%)')

            out_png_summary = base_out / f'{season}_{team}_summary.png'
            # Do not call tight_layout here; we intentionally reserved top margin
            fig.savefig(out_png_summary, dpi=150)
            plt.close(fig)
        except Exception as e:
            print('failed to create combined summary plot for', team, e)
            out_png_summary = None

        # Calculate Goals
        # team_xg and opp_xg are already computed above
        team_goals = 0
        opp_goals = 0
        try:
            goals_df = df_games[df_games['event'].astype(str).str.strip().str.lower() == 'goal']
            if not goals_df.empty:
                # Reuse calculate_shot_attempts logic or simple mask
                # We need to identify team vs opp
                # calculate_shot_attempts logic:
                tupper = str(team).strip().upper()
                def _is_team_goal(r):
                    try:
                        home_abb = r.get('home_abb')
                        away_abb = r.get('away_abb')
                        if home_abb is not None and str(home_abb).upper() == tupper:
                            return str(r.get('team_id')) == str(r.get('home_id'))
                        if away_abb is not None and str(away_abb).upper() == tupper:
                            return str(r.get('team_id')) == str(r.get('away_id'))
                        return False
                    except: return False
                
                is_team = goals_df.apply(_is_team_goal, axis=1)
                team_goals = int(is_team.sum())
                opp_goals = int((~is_team).sum())
        except Exception as e:
            print(f"season_analysis: error calculating goals for {team}: {e}")

        # save JSON summary
        summary = {
            'team': team,
            'n_events': n_events,
            'team_xg': team_xg,
            'opp_xg': opp_xg,
            'team_seconds': team_seconds,
            'team_xg_per60': (team_xg / team_seconds * 3600.0) if team_seconds > 0 else None,
            'opp_xg_per60': (opp_xg / team_seconds * 3600.0) if team_seconds > 0 else None,
            'team_goals': team_goals,
            'opp_goals': opp_goals,
            'team_attempts': 0,
            'opp_attempts': 0,
            'out_png_summary': str(out_png_summary) if out_png_summary is not None else None,
        }
        # write JSON
        try:
            out_json = base_out / f'{season}_{team}_summary.json'
            with open(out_json, 'w') as fh:
                json.dump(summary, fh, indent=2)
        except Exception:
            pass

        results['teams'][team] = summary

        # Append to lists for summary plot
        if summary.get('team_xg_per60') is not None and summary.get('opp_xg_per60') is not None:
            all_team_names.append(team)
            all_team_xg_per60.append(summary['team_xg_per60'])
            all_opp_xg_per60.append(summary['opp_xg_per60'])

    # Generate Season Summary Plot: xG For vs xG Against
    if all_team_names:
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Scatter plot
            ax.scatter(all_team_xg_per60, all_opp_xg_per60, alpha=0.7, c='blue', edgecolors='k')
            
            # Add labels
            for i, txt in enumerate(all_team_names):
                ax.annotate(txt, (all_team_xg_per60[i], all_opp_xg_per60[i]), 
                            xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            # Unity line
            try:
                min_val = min(min(all_team_xg_per60), min(all_opp_xg_per60))
                max_val = max(max(all_team_xg_per60), max(all_opp_xg_per60))
                margin = (max_val - min_val) * 0.1
                line_min = max(0, min_val - margin)
                line_max = max_val + margin
                ax.plot([line_min, line_max], [line_min, line_max], 'k--', alpha=0.5, label='Even')
                
                # Set limits to be square-ish if possible, or at least cover the data
                ax.set_xlim(line_min, line_max)
                ax.set_ylim(line_min, line_max)
            except Exception:
                pass
            
            # Labels and Title
            ax.set_xlabel('xG For / 60')
            ax.set_ylabel('xG Against / 60')
            ax.set_title(f'Season Analysis {season}: xG Performance')
            ax.grid(True, linestyle=':', alpha=0.6)
            
            out_plot = base_out / f'{season}_season_comparison.png'
            fig.savefig(out_plot, dpi=150)
            plt.close(fig)
            print(f"Saved season comparison plot to {out_plot}")
        except Exception as e:
            print(f"Failed to generate season comparison plot: {e}")

    # Calculate percentiles and save combined team summary
    try:
        # Helper for percentile calculation
        def _calc_pct(val, all_vals):
            if not all_vals or val is None:
                return None
            k = sum(1 for v in all_vals if v < val)
            eq = sum(1 for v in all_vals if v == val)
            return (k + 0.5 * eq) / len(all_vals) * 100.0

        # Extract lists for percentile calculation
        xg_for_vals = [s['team_xg_per60'] for s in results['teams'].values() if s.get('team_xg_per60') is not None]
        xg_ag_vals = [s['opp_xg_per60'] for s in results['teams'].values() if s.get('opp_xg_per60') is not None]
        
        combined_summaries = []
        for team, summary in results['teams'].items():
            # Calculate percentiles
            summary['off_percentile'] = _calc_pct(summary.get('team_xg_per60'), xg_for_vals)
            # For xGA, higher percentile means MORE xGA (bad defense). 
            # We stick to raw percentile of the metric.
            summary['def_percentile'] = _calc_pct(summary.get('opp_xg_per60'), xg_ag_vals)
            
            combined_summaries.append(summary)
            
            # Update individual JSON with percentiles
            try:
                out_json = base_out / f'{season}_{team}_summary.json'
                with open(out_json, 'w') as fh:
                    json.dump(summary, fh, indent=2)
            except Exception:
                pass

        # Save combined summary
        out_combined_json = base_out / f'{season}_team_summary.json'
        with open(out_combined_json, 'w') as fh:
            json.dump(combined_summaries, fh, indent=2)
            
    except Exception as e:
        print(f"Failed to calculate percentiles or save combined summary: {e}")

    # save a small league summary JSON
    try:
        out_league_json = base_out / f'{season}_league_summary.json'
        with open(out_league_json, 'w') as fh:
            json.dump(results['league'], fh, indent=2)
    except Exception:
        pass

    return results


if __name__ == '__main__':
    """Command-line interface for xG analysis and mapping.

    This script provides tools for generating xG heatmaps and analyzing team performance
    relative to league baselines. It supports single-game analysis, full-season analysis,
    and league baseline computation.

    Usage Examples:

    1. Generate a standard xG map for a specific team and season:
       python analyze.py --season 20252026 --team PHI

    2. Generate an xG map for a single game (using game ID):
       python analyze.py --game-id 2025020339

    3. Generate an xG map for a single game with specific conditions (e.g., 5v5 only):
       python analyze.py --game-id 2025020339 --condition '{"game_state": "5v5"}'

    4. Generate an xG map for a single game with NO filtering (empty condition):
       python analyze.py --game-id 2025020339 --condition '{}'

    5. Run a full season analysis for all teams (relative to league baseline):
       python analyze.py --season 20252026 --season-analysis

    6. Compute a new league baseline for a season:
       python analyze.py --season 20252026 --league-baseline --baseline-mode compute

    7. Run analysis for a specific team with custom output path and orientation:
       python analyze.py --season 20252026 --team PHI --out static/PHI_custom.png --orient-all-left

    Notes:
      - The `--condition` argument accepts a JSON string to filter events.
      - The `--season-analysis` flag runs the comprehensive workflow used for generating
        relative performance maps and summary statistics.
    """

    import argparse
    import json

    parser = argparse.ArgumentParser(prog='analyze.py', description='Run xgs_map for a season and team (example).')
    parser.add_argument('--season', default='20252026', help='Season string (e.g. 20252026)')
    parser.add_argument('--team', required=False, help='Team abbreviation or id to filter (e.g. PHI)')
    parser.add_argument('--out', default='static/xg_map_example.png', help='Output image path')
    parser.add_argument('--orient-all-left', action='store_true', help='Orient all shots to the left')
    parser.add_argument('--behavior', choices=['load', 'train'], default='load', help='Classifier behavior for xg model (load or train)')
    # by default we return heatmaps; use --no-heatmaps to disable
    parser.add_argument('--no-heatmaps', dest='return_heatmaps', action='store_false', help='Do not return heatmaps (enabled by default)')
    parser.set_defaults(return_heatmaps=True)
    parser.add_argument('--csv-path', default=None, help='Optional explicit CSV path to use (overrides season search)')
    parser.add_argument('--condition', default=None, help='Optional JSON condition string (e.g. \'{"game_state": "5v5"}\')')
    # NEW: quick-run flag to generate per-team xG pct maps for the whole season
    parser.add_argument('--run-all', action='store_true', help='Run full-season xG maps for all teams')
    # NEW: allow specifying a single game id to generate an xG map for that game
    parser.add_argument('--game-id', default=None, help='Game ID (e.g. 2025020339) to generate an xG map for a single game')
    # NEW: league baseline options
    parser.add_argument('--league-baseline', action='store_true', help='Compute or load league baseline xG map')
    parser.add_argument('--baseline-mode', choices=['compute', 'load'], default='load', 
                       help='Mode for league baseline: compute (calculate from data) or load (read from disk)')
    # NEW: season analysis option
    parser.add_argument('--season-analysis', action='store_true', 
                       help='Run unified season analysis: compute per-team maps relative to league baseline')
    parser.add_argument('--no-cache', action='store_true', help='Disable API caching')

    args = parser.parse_args()

    if args.no_cache:
        from . import nhl_api
        nhl_api.set_caching(False)
        print("API caching disabled.")

    # Parse condition if provided
    parsed_condition = None
    if args.condition:
        try:
            parsed_condition = json.loads(args.condition)
        except Exception as e:
            print(f"Error parsing condition JSON: {e}")
            sys.exit(1)
    
    # Execute based on arguments - only runs when called as main script
    if args.league_baseline:
        # Compute or load league baseline
        print(f"Running league baseline analysis (mode={args.baseline_mode})")
        result = league(season=args.season, csv_path=args.csv_path, mode=args.baseline_mode, condition=parsed_condition)
        print("\nLeague baseline stats:")
        print(json.dumps(result.get('stats', {}), indent=2))
    elif args.season_analysis:
        # Run unified season analysis
        print(f"Running unified season analysis for {args.season}")
        result = season_analysis(
            season=args.season,
            csv_path=args.csv_path,
            baseline_mode=args.baseline_mode,
            condition=parsed_condition,
        )
        print("\nSeason analysis complete!")
        if 'summary_table' in result and not result['summary_table'].empty:
            print("\nTop 10 teams by relative xG/60:")
            print(result['summary_table'].head(10).to_string(index=False))
    elif args.run_all:
        # Use the new unified season_analysis routine instead of xg_maps_for_season
        print(f"Running season analysis for all teams (season={args.season})")
        result = season_analysis(
            season=args.season,
            csv_path=args.csv_path,
            baseline_mode=args.baseline_mode,
            condition=parsed_condition,
        )
        print("\nSeason analysis complete!")
        if 'summary_table' in result and not result['summary_table'].empty:
            print("\nTeam rankings by relative xG/60:")
            print(result['summary_table'].to_string(index=False))
    elif args.team:
        if parsed_condition:
            condition = parsed_condition.copy()
            # Ensure team is in condition if not explicitly provided in JSON
            if 'team' not in condition:
                condition['team'] = args.team
        else:
            condition = {'game_state': ['5v5'], 'is_net_empty': [0], 'team': args.team}
        xgs_map(season=args.season, condition=condition, behavior=args.behavior, 
                out_path=args.out, orient_all_left=args.orient_all_left,
                return_heatmaps=args.return_heatmaps, csv_path=args.csv_path)
    elif args.game_id:
        # Run xgs_map for a single game id. We pass the game_id explicitly.
        # Use provided condition if available, otherwise default to 5v5 + net not empty.
        if parsed_condition is not None:
            condition = parsed_condition.copy()
        else:
            condition = {'game_state': ['5v5'], 'is_net_empty': [0]}
            
        # If output path is the default, change it to use the game_id
        out_path_arg = args.out
        if out_path_arg == 'static/xg_map_example.png':
            out_path_arg = f'static/{args.game_id}.png'
            
        # Force returning full outputs for a single-game run so the CLI
        # user can inspect heatmaps, filtered dataframe, and summary stats.
        try:
            out_path_ret, heatmaps_ret, filtered_df_ret, summary_stats_ret = xgs_map(
                season=args.season,
                game_id=args.game_id,
                condition=condition,
                behavior=args.behavior,
                out_path=out_path_arg,
                orient_all_left=args.orient_all_left,
                # ensure we get the detailed outputs for inspection
                return_heatmaps=True,
                return_filtered_df=True,
                csv_path=args.csv_path,
            )
            print(f"xgs_map completed for game_id={args.game_id}; out_path={out_path_ret}")
            try:
                # report heatmaps summary (may be dict/tuple/ndarray)
                if heatmaps_ret is None:
                    print('heatmaps: None')
                else:
                    try:
                        import numpy as _np
                        if isinstance(heatmaps_ret, dict):
                            print('heatmaps keys:', list(heatmaps_ret.keys()))
                            for k, v in heatmaps_ret.items():
                                try:
                                    print(f"  {k}: type={type(v)}, shape={getattr(v, 'shape', None)}")
                                except Exception:
                                    print(f"  {k}: type={type(v)}")
                        elif isinstance(heatmaps_ret, (_np.ndarray, list, tuple)):
                            print('heatmaps: ndarray/list/tuple, info ->', type(heatmaps_ret), getattr(heatmaps_ret, 'shape', None))
                        else:
                            print('heatmaps:', type(heatmaps_ret))
                    except Exception:
                        print('heatmaps (repr):', repr(heatmaps_ret))
            except Exception:
                print('Could not introspect heatmaps')

            # report filtered dataframe
            if filtered_df_ret is None:
                print('filtered_df: None')
            else:
                try:
                    print(f"filtered_df rows={getattr(filtered_df_ret, 'shape', None)}")
                    # attempt to persist for later inspection
                    try:
                        import os
                        out_dir = 'static'
                        os.makedirs(out_dir, exist_ok=True)
                        csv_out = os.path.join(out_dir, f'{args.game_id}_filtered.csv')
                        filtered_df_ret.to_csv(csv_out, index=False)
                        print(f'Wrote filtered dataframe to {csv_out}')
                    except Exception as e:
                        print('Failed to save filtered dataframe:', e)
                except Exception:
                    print('filtered_df (repr):', repr(filtered_df_ret))

            # print summary stats if present
            try:
                print('summary_stats:', summary_stats_ret)
            except Exception:
                pass
        except Exception as e:
            print('xgs_map for single game failed:', e)


def generate_special_teams_plot(season, teams, out_dir):
    """
    Generates a combined Special Teams plot for each team.
    Left half: 5v4 Offense (from 5v4 relative map)
    Right half: 4v5 Defense (from 4v5 relative map)
    
    Also generates a combined summary JSON and scatter plot.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from .rink import draw_rink
    from .plot import add_summary_text
    import json
    # generate_scatter_plot is in this module

    if teams is None:
        try:
            with open('static/teams.json', 'r') as f:
                teams_data = json.load(f)
            teams = [t.get('abbr') for t in teams_data if 'abbr' in t]
        except Exception as e:
            print(f"generate_special_teams_plot: failed to load teams.json: {e}")
            return

    print(f"Generating Special Teams plots for {len(teams)} teams...")
    
    # Ensure output directory exists
    st_out_dir = os.path.join(out_dir, 'SpecialTeams')
    os.makedirs(st_out_dir, exist_ok=True)
    
    # Paths to source data
    # out_dir is typically static/league/20252026
    
    base_dir = out_dir
    dir_5v4 = os.path.join(base_dir, '5v4')
    dir_4v5 = os.path.join(base_dir, '4v5')
    
    # Load summary stats for 5v4 and 4v5 to get rankings/stats
    stats_5v4 = []
    stats_4v5 = []
    
    try:
        with open(os.path.join(dir_5v4, f'{season}_team_summary.json'), 'r') as f:
            stats_5v4 = json.load(f)
    except Exception:
        print("Warning: Could not load 5v4 summary stats")

    try:
        with open(os.path.join(dir_4v5, f'{season}_team_summary.json'), 'r') as f:
            stats_4v5 = json.load(f)
    except Exception:
        print("Warning: Could not load 4v5 summary stats")

    # Helper to find team stats in list
    def get_team_stats(stats_list, team_abbr):
        if isinstance(stats_list, list):
            for s in stats_list:
                if s.get('team') == team_abbr:
                    return s
        return {}

    combined_summary = []

    for team in teams:
        # Load maps
        map_5v4_path = os.path.join(dir_5v4, f'{team}_relative_combined.npy')
        map_4v5_path = os.path.join(dir_4v5, f'{team}_relative_combined.npy')
        
        # Get stats even if map doesn't exist (for table)
        t_stats_5v4 = get_team_stats(stats_5v4, team)
        t_stats_4v5 = get_team_stats(stats_4v5, team)
        
        # Create combined stats entry
        # We want PP Offense (team_xg_per60 from 5v4) vs PK Defense (other_xg_per60 from 4v5)
        # Note: 4v5 'other_xg_per60' is xGA/60 (shots against team while team is PK)
        
        pp_xg_per60 = t_stats_5v4.get('team_xg_per60', 0.0)
        pk_xga_per60 = t_stats_4v5.get('other_xg_per60', 0.0)
        
        # Also sum raw counts for table
        pp_goals = t_stats_5v4.get('team_goals', 0)
        pk_goals_against = t_stats_4v5.get('other_goals', 0)
        
        pp_xgs = t_stats_5v4.get('team_xgs', 0.0)
        pk_xgs_against = t_stats_4v5.get('other_xgs', 0.0)
        
        pp_attempts = t_stats_5v4.get('team_attempts', 0)
        pk_attempts_against = t_stats_4v5.get('other_attempts', 0)
        
        pp_toi = t_stats_5v4.get('team_seconds', 0.0)
        pk_toi = t_stats_4v5.get('team_seconds', 0.0) # PK TOI
        
        # Net Special Teams xG? (PP xGF - PK xGA) - crude metric but maybe useful
        net_st_xg = pp_xgs - pk_xgs_against
        
        combined_entry = {
            'team': team,
            'Team': team, # For consistency with other summaries
            'n_games': t_stats_5v4.get('n_games', 0), # Assume same games
            
            # PP Stats (Offense)
            'team_goals': pp_goals,
            'team_xgs': pp_xgs,
            'team_attempts': pp_attempts,
            'team_seconds': pp_toi,
            'team_xg_per60': pp_xg_per60, # This will be xGF/60 in scatter
            
            # PK Stats (Defense) - mapped to 'other' fields for scatter compatibility
            'other_goals': pk_goals_against,
            'other_xgs': pk_xgs_against,
            'other_attempts': pk_attempts_against,
            'other_seconds': pk_toi,
            'other_xg_per60': pk_xga_per60, # This will be xGA/60 in scatter
            
            # Extra
            'net_st_xg': net_st_xg
        }
        combined_summary.append(combined_entry)

        if not os.path.exists(map_5v4_path) or not os.path.exists(map_4v5_path):
            # print(f"Skipping {team}: missing 5v4 or 4v5 map")
            continue
            
        try:
            map_5v4 = np.load(map_5v4_path)
            map_4v5 = np.load(map_4v5_path)
            
            # Stitch maps
            # Grid assumptions: 200 x 85 (from histogram2d)
            # x indices: 0 to 199.
            # Midpoint is 100.
            
            combined_st = np.full_like(map_5v4, np.nan)
            
            mid = map_5v4.shape[1] // 2 # 100
            
            # Left half from 5v4 (Offense)
            combined_st[:, :mid] = map_5v4[:, :mid]
            
            # Right half from 4v5 (Defense)
            combined_st[:, mid:] = map_4v5[:, mid:]
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 5))
            draw_rink(ax=ax)
            
            # Extent
            gx = np.arange(-100.0, 100.0 + 1.0, 1.0)
            gy = np.arange(-42.5, 42.5 + 1.0, 1.0)
            extent = (gx[0] - 0.5, gx[-1] + 0.5, gy[0] - 0.5, gy[-1] + 0.5)
            
            # Use SymLogNorm for diff metric to enhance contrast (same as 5v5)
            from matplotlib.colors import SymLogNorm
            vmax = 0.0006
            norm = SymLogNorm(linthresh=1e-5, linscale=1.0, vmin=-vmax, vmax=vmax, base=10)
            
            cmap = plt.get_cmap('RdBu_r')
            try:
                cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))
            except:
                pass
                
            m = np.ma.masked_invalid(combined_st)
            im = ax.imshow(m, extent=extent, origin='lower', cmap=cmap, norm=norm)
            
            # Colorbar with human readable labels
            cbar_ticks = [-0.0006, -0.0001, -0.00001, 0, 0.00001, 0.0001, 0.0006]
            cbar_ticklabels = ['High -', 'Med -', 'Low -', 'Avg', 'Low +', 'Med +', 'High +']
            
            cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
            cbar.set_label('Relative xG/60 Difference', rotation=270, labelpad=20)
            cbar.set_ticks(cbar_ticks)
            cbar.ax.set_yticklabels(cbar_ticklabels)
            
            # Summary Text
            # Use the combined stats we just built
            text_stats = combined_entry.copy()
            text_stats['home_xg'] = text_stats['team_xgs']
            text_stats['away_xg'] = text_stats['other_xgs']
            text_stats['have_xg'] = True
            text_stats['home_goals'] = text_stats['team_goals']
            text_stats['away_goals'] = text_stats['other_goals']
            text_stats['home_attempts'] = text_stats['team_attempts']
            text_stats['away_attempts'] = text_stats['other_attempts']
            
            # Add summary text
            add_summary_text(
                ax=ax,
                stats=text_stats,
                main_title=f"{team} Special Teams (PP Off / PK Def)",
                is_season_summary=True,
                team_name=team,
                filter_str="Special Teams"
            )
            
            out_path = os.path.join(st_out_dir, f'{team}_special_teams_map.png')
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            print(f"generate_special_teams_plot: error for {team}: {e}")
            import traceback
            traceback.print_exc()

    # Save Combined Summary
    summary_path = os.path.join(st_out_dir, f'{season}_team_summary.json')
    try:
        with open(summary_path, 'w') as f:
            json.dump(combined_summary, f, indent=2)
        print(f"Saved Special Teams summary to {summary_path}")
    except Exception as e:
        print(f"Failed to save Special Teams summary: {e}")

    # Generate Scatter Plot
    # generate_scatter_plot expects list of dicts with 'team_xg_per60' and 'other_xg_per60'
    # We have mapped PP Off -> team_xg_per60 and PK Def -> other_xg_per60
    print("Generating Special Teams scatter plot...")
    try:
        generate_scatter_plot(combined_summary, st_out_dir, condition_name="SpecialTeams")
    except Exception as e:
        print(f"Failed to generate Special Teams scatter plot: {e}")


def orient_all(df_in, target: str = 'left', selected_team: object = None, selected_role: str = 'team'):
    """Return a copy of df_in with x_a/y_a rotated according to either a fixed target
    ('left' or 'right') or relative to a `selected_team`.

    If `selected_team` is provided, rows where the shooter matches
    `selected_team` will be oriented to the left and all other rows to the right. If `selected_team` is None, behavior falls back to the simpler
    `target`-based orientation (all left or all right).
    """
    df2 = df_in.copy()
    import pandas as pd

    # determine goal x positions (try to use helper from plot/rink if available)
    try:
        from .plot import rink_goal_xs
        left_goal_x, right_goal_x = rink_goal_xs()
    except Exception:
        # sensible defaults (distance from center to goal line)
        left_goal_x, right_goal_x = -89.0, 89.0

    def attacked_goal_x_for_row(r):
        # replicate logic from plot.adjust_xy_for_homeaway
        try:
            team_id = r.get('team_id')
            home_id = r.get('home_id')
            home_def = r.get('home_team_defending_side')
            if pd.isna(team_id) or pd.isna(home_id):
                return right_goal_x
            if str(team_id) == str(home_id):
                # shooter is home: they attack opposite of home's defended side
                if home_def == 'left':
                    return right_goal_x
                elif home_def == 'right':
                    return left_goal_x
                else:
                    return right_goal_x
            else:
                if home_def == 'left':
                    return left_goal_x
                elif home_def == 'right':
                    return right_goal_x
                else:
                    return left_goal_x
        except Exception:
            return right_goal_x

    # compute attacked goal series
    attacked = df2.apply(attacked_goal_x_for_row, axis=1)

    # determine desired goal per-row
    if selected_team is not None:
        tstr = str(selected_team).strip()
        tid = None
        try:
            tid = int(tstr)
        except Exception:
            tid = None

        def is_selected(row):
            try:
                if tid is not None:
                    return str(row.get('team_id')) == str(tid)
                shooter_id = row.get('team_id')
                if pd.isna(shooter_id):
                    return False
                if row.get('home_id') is not None and str(row.get('home_id')) == str(row.get('team_id')):
                    # home team membership
                    if row.get('home_abb') is not None and str(row.get('home_abb')).upper() == tstr.upper():
                        return True
                if row.get('away_id') is not None and str(row.get('away_id')) == str(row.get('team_id')):
                    if row.get('away_abb') is not None and str(row.get('away_abb')).upper() == tstr.upper():
                        return True
                # fallback: compare abbreviations if present
                if row.get('home_abb') is not None and str(row.get('home_abb')).upper() == tstr.upper() and str(row.get('team_id')) == str(row.get('home_id')):
                    return True
                if row.get('away_abb') is not None and str(row.get('away_abb')).upper() == tstr.upper() and str(row.get('team_id')) == str(row.get('away_id')):
                    return True
            except Exception:
                return False
            return False

        # selected_role determines which rows are treated as the "selected"
        # group. If selected_role == 'team', rows matching selected_team are
        # oriented to the left and others to the right. If
        # selected_role == 'other', the logic is flipped: rows matching
        # selected_team are oriented to the right and others to the left.
        if selected_role == 'team':
            desired = df2.apply(lambda r: left_goal_x if is_selected(r) else right_goal_x, axis=1)
        else:
            desired = df2.apply(lambda r: right_goal_x if is_selected(r) else left_goal_x, axis=1)
    else:
        desired = left_goal_x if target == 'left' else right_goal_x

    # produce adjusted coords
    xcol = 'x'
    ycol = 'y'
    df2['x_a'] = df2.get('x')
    df2['y_a'] = df2.get('y')
    mask = (attacked != desired) & df2[xcol].notna() & df2[ycol].notna()
    try:
        df2.loc[mask, ['x_a', 'y_a']] = -df2.loc[mask, [xcol, ycol]].values
    except Exception:
        # fallback loop if vectorized assignment fails
        for idx in df2.loc[mask].index:
            try:
                df2.at[idx, 'x_a'] = -float(df2.at[idx, xcol])
                df2.at[idx, 'y_a'] = -float(df2.at[idx, ycol])
            except Exception:
                df2.at[idx, 'x_a'] = df2.at[idx, xcol]
                df2.at[idx, 'y_a'] = df2.at[idx, ycol]
    return df2


def rotate_heat_180(heat):
    import numpy as _np
    if heat is None:
        return None
    return _np.flipud(_np.fliplr(heat))
