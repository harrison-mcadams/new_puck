"""Minimal Flask web UI for the demo shot-plot.

This module provides two simple routes:
- GET / : render a page that displays the currently-generated static plot
  saved at `static/shot_plot.png`.
- POST /replot : trigger regeneration of the plot by calling into
  `plot_game.plot_shots` and then redirecting back to the index page.

Notes:
- The current /replot implementation calls the plotting function *synchronously*,
  which means the HTTP request blocks until the plot is generated. For long
  running plotting tasks you should move that work into a background job queue
  (RQ/Celery) or a separate worker process and return a quick response to the
  client instead.
"""

from flask import Flask, render_template, redirect, url_for, request, send_from_directory
import os
import pandas as pd
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


app = Flask(__name__, static_folder="web/static", template_folder="web/templates")
ANALYSIS_DIR = os.path.abspath("analysis")
LOG_DIR = os.path.abspath("logs")

@app.route('/analysis/<path:filename>')
def analysis_file(filename):
    return send_from_directory(ANALYSIS_DIR, filename)

# Small default team list fallback (abbr,name). This is used when the NHL
# schedule API is unavailable or returns 'access denied' so the UI remains
# functional for manual Game ID entry.
DEFAULT_TEAMS = [
    {'abbr': 'BOS', 'name': 'Bruins'},
    {'abbr': 'NYR', 'name': 'Rangers'},
    {'abbr': 'PHI', 'name': 'Flyers'},
    {'abbr': 'PIT', 'name': 'Penguins'},
    {'abbr': 'TOR', 'name': 'Maple Leafs'},
    {'abbr': 'CHI', 'name': 'Blackhawks'},
]


# Simple in-memory cache to avoid repeated expensive API calls during a dev
# session. Keys map to (timestamp, payload). TTL in seconds.
_CACHE = {}
_CACHE_TTL = 300
# Store metadata about the last loaded game to populate UI filters
_LAST_GAME_METADATA = {}


def _cache_get(key):
    import time
    ent = _CACHE.get(key)
    if not ent:
        return None
    ts, payload = ent
    if time.time() - ts > _CACHE_TTL:
        try:
            del _CACHE[key]
        except Exception:
            pass
        return None
    return payload


def _cache_set(key, payload):
    import time
    _CACHE[key] = (time.time(), payload)


@app.route("/")
def index():
    """Render the main page showing the current shot plot if it exists."""
    img_filename = "shot_plot.png"
    img_path = os.path.join(ANALYSIS_DIR, img_filename)
    exists = os.path.exists(img_path)
    
    # Pass metadata to template
    return render_template("index.html", exists=exists, img=img_filename, metadata=_LAST_GAME_METADATA)


@app.route("/replot", methods=["POST"])
def replot():
    """Trigger re-generation of the plot and redirect back to index.

    Accepts form fields:
    - game: Game ID
    - game_state: list of game states (e.g. '5v5')
    - is_net_empty: '0', '1', or '' (any)
    - player_id: Player ID (int) or '' (any)
    - condition: JSON string (fallback/legacy)
    """
    try:
        from puck import plot
        from puck import fit_xgs
        from puck import analyze
        import json
        import numpy as np
    except Exception as e:
        return (f"plotting modules not available: {e}", 500)

    # Log the received form keys for debugging at debug level
    try:
        logger.debug('replot: received form keys: %s', {k: request.form.get(k) for k in request.form.keys()})
    except Exception:
        logger.debug('replot: failed to enumerate form keys', exc_info=True)

    raw_game = (request.form.get('game') or '').strip()
    if not raw_game:
        logger.debug('replot: no game provided in form; analyze_game will choose default')
        game_val = None
    else:
        cleaned = raw_game.replace(' ', '')
        if cleaned.isdigit():
            try:
                game_val = int(cleaned)
            except Exception:
                game_val = cleaned
        else:
            game_val = cleaned
        logger.debug("replot: received game field '%s' -> normalized '%s'", raw_game, game_val)

    # Construct condition from form fields
    condition = {}
    
    # 1. Game State (Multi-select)
    game_states = request.form.getlist('game_state')
    if game_states:
        # If "Any" (empty string) is selected, it overrides specific selections -> No Filter
        if '' in game_states:
            pass
        else:
            # Filter out empty strings and set condition
            valid_states = [s for s in game_states if s.strip()]
            if valid_states:
                condition['game_state'] = valid_states

    # 2. Net Empty
    net_empty = request.form.get('is_net_empty')
    if net_empty is not None and net_empty.strip() != '':
        try:
            condition['is_net_empty'] = [int(net_empty)]
        except Exception:
            pass

    # 3. Player ID
    player_id = request.form.get('player_id')
    if player_id and player_id.strip():
        try:
            condition['player_id'] = int(player_id)
        except Exception:
            pass

    # 4. JSON Fallback (if no specific fields provided, or to augment?)
    # If condition is still empty, check the JSON field
    if not condition:
        raw_condition = (request.form.get('condition') or '{}').strip()
        try:
            if raw_condition:
                loaded = json.loads(raw_condition)
                if isinstance(loaded, dict):
                    condition = loaded
        except Exception as e:
            logger.warning(f"Failed to parse condition JSON: {e}")

    out_path = os.path.join(ANALYSIS_DIR, 'shot_plot.png')

    try:
        # Use analyze.xgs_map which handles filtering and plotting
        # We pass game_id (if any) and the condition.
        # Note: xgs_map expects game_id as a keyword argument.
        ret = analyze.xgs_map(
            game_id=game_val,
            condition=condition,
            out_path=out_path,
            show=False,
            # Ensure we don't try to open a window
            return_heatmaps=False,
            # Default behavior: show only shots on goal and goals, plus the xG heatmap
            events_to_plot=['shot-on-goal', 'goal', 'xgs'],
            return_filtered_df=True,
            force_refresh=True
        )
        
        # Explicitly close the figure to prevent memory leaks and plot stacking
        # in the persistent web server process.
        # xgs_map returns (out_path, heatmaps, df, summary_stats)
        # It closes the figure internally now (via my fix in analyze.py), so we don't strictly need to here,
        # but the return signature is different from what I thought earlier.
        # Earlier I thought it returned (fig, ax).
        # But analyze.py returns (out_path, ret_heat, ret_df, summary_stats).
        # So ret[0] is out_path. plt.close(ret[0]) would be wrong.
        # Since I fixed analyze.py to close the figure, I can remove the plt.close logic here or just call plt.close('all').
        plt.close('all')

        # Extract metadata from returned DataFrame (ret[2])
        if isinstance(ret, tuple) and len(ret) >= 3:
            df = ret[2]
            if df is not None and not df.empty:
                # Extract unique game states
                states = sorted(df['game_state'].dropna().unique().tolist())
                
                # Extract players (id and name)
                players = []
                if 'player_id' in df.columns:
                    # Check if player_name exists
                    has_name = 'player_name' in df.columns
                    # Get unique pairs
                    if has_name:
                        pairs = df[['player_id', 'player_name']].dropna().drop_duplicates()
                        for _, row in pairs.iterrows():
                            players.append({'id': int(row['player_id']), 'name': row['player_name']})
                    else:
                        # Just IDs
                        pids = df['player_id'].dropna().unique()
                        for pid in pids:
                            players.append({'id': int(pid), 'name': f"Player {int(pid)}"})
                
                # Sort players by name
                players.sort(key=lambda x: x['name'])
                
                # Update global metadata
                global _LAST_GAME_METADATA
                _LAST_GAME_METADATA = {
                    'game_id': game_val,
                    'game_states': states,
                    'players': players,
                    'current_condition': condition
                }

    except Exception as e:
        logger.exception('Unhandled error in replot handler: %s', e)
        return (f"Error generating plot: {e}", 500)

    return redirect(url_for('index'))


@app.route("/league_stats")
def league_stats():
    """Render the league statistics page."""
    import json
    
    # Get query parameters
    season = request.args.get('season', '20252026')
    game_state = request.args.get('game_state', '5v5')
    
    # Load summary from static/league/{season}/{game_state}/{season}_team_summary.json
    # Note: run_league_stats.py saves to static/league/{season}/{game_state}
    # Special Teams might not have a summary json, or it might be different.
    # If game_state is SpecialTeams, we might expect no summary or a specific one.
    # Currently run_league_stats.py generates special teams plots but maybe not a summary table?
    # It calls analyze.generate_special_teams_plot.
    
    summary_path = os.path.join(ANALYSIS_DIR, "league", season, game_state, f"{season}_team_summary.json")
    scatter_path = f"league/{season}/{game_state}/scatter.png"
    
    # Special Teams override for scatter path (it might not exist, or be different)
    # if game_state == 'SpecialTeams':
    #     # Special Teams doesn't have a scatter plot generated by default in run_league_stats.py
    #     scatter_path = "" 
    
    stats = []
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r') as f:
                raw_stats = json.load(f)
                
            # Transform stats for display
            for row in raw_stats:
                if row.get('team') == 'League':
                    continue
                    
                # Calculate percentages
                gf = row.get('team_goals', 0)
                ga = row.get('other_goals', 0)
                g_total = gf + ga
                gf_pct = (gf / g_total * 100) if g_total > 0 else 0.0
                
                xgf = row.get('team_xgs', 0)
                xga = row.get('other_xgs', 0)
                xg_total = xgf + xga
                xgf_pct = (xgf / xg_total * 100) if xg_total > 0 else 0.0
                
                cf = row.get('team_attempts', 0)
                ca = row.get('other_attempts', 0)
                c_total = cf + ca
                cf_pct = (cf / c_total * 100) if c_total > 0 else 0.0
                
                toi_sec = row.get('team_seconds', 0)
                toi_min = int(toi_sec / 60)
                
                stats.append({
                    'Team': row.get('team'),
                    'GP': row.get('n_games'),
                    'TOI': toi_min,
                    'GF': gf,
                    'GA': ga,
                    'GF%': round(gf_pct, 1),
                    'xGF': round(xgf, 1),
                    'xGA': round(xga, 1),
                    'xGF%': round(xgf_pct, 1),
                    'CF': cf,
                    'CA': ca,
                    'CF%': round(cf_pct, 1),
                    'xGF/60': round(row.get('team_xg_per60', 0), 2),
                    'xGA/60': round(row.get('other_xg_per60', 0), 2)
                })
                
            # Sort by xGF% descending by default
            stats.sort(key=lambda x: x['xGF%'], reverse=True)
                
        except Exception as e:
            logger.error(f"Failed to load league stats: {e}")
            
    return render_template("league_stats.html", stats=stats, season=season, game_state=game_state, scatter_img=scatter_path)


@app.route("/team_maps")
def team_maps():
    """Render the gallery of team relative maps."""
    import glob
    import json
    
    # Get query parameters
    season = request.args.get('season', '20252026')
    game_state = request.args.get('game_state', '5v5')
    
    # Directory where maps are stored
    maps_dir = os.path.join(ANALYSIS_DIR, "league", season, game_state)
    
    teams_data = []
    
    # We can use the summary json to get the list of teams, or just glob the files.
    # Summary json is safer to get the list of valid teams.
    summary_path = os.path.join(maps_dir, f"{season}_team_summary.json")
    
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r') as f:
                summary = json.load(f)
                # Filter out 'League' entry if present
                teams = sorted([row['team'] for row in summary if row.get('team') != 'League'])
                
                for team in teams:
                    # Check if map exists
                    # Map filename: {team}_relative_map.png
                    # OR for SpecialTeams: {team}_special_teams_map.png
                    
                    if game_state == 'SpecialTeams':
                        map_filename = f"{team}_special_teams_map.png"
                    else:
                        map_filename = f"{team}_relative.png"
                        
                    map_path = os.path.join(maps_dir, map_filename)
                    
                    if os.path.exists(map_path):
                        teams_data.append({
                            'name': team,
                            'map_url': f"league/{season}/{game_state}/{map_filename}"
                        })
        except Exception as e:
            logger.error(f"Failed to load teams for maps: {e}")
            
    # Fallback for SpecialTeams if summary doesn't exist to provide team list
    if not teams_data and game_state == 'SpecialTeams':
        # Glob for files
        search_path = os.path.join(maps_dir, "*_special_teams_map.png")
        files = glob.glob(search_path)
        for fpath in files:
            fname = os.path.basename(fpath)
            # Extract team name: {team}_special_teams_map.png
            team_name = fname.replace("_special_teams_map.png", "")
            teams_data.append({
                'name': team_name,
                'map_url': f"league/{season}/{game_state}/{fname}"
            })
        teams_data.sort(key=lambda x: x['name'])
    
    return render_template("team_maps.html", teams=teams_data, season=season, game_state=game_state)


@app.route("/team_stats/<season>/<game_state>/<team>")
def team_stats(season, game_state, team):
    """Render the team statistics page with relative map."""
    
    # Construct paths
    # Construct paths
    # Map is at static/league/{season}/{game_state}/{team}_relative_map.png
    # OR {team}_special_teams_map.png for SpecialTeams
    
    if game_state == 'SpecialTeams':
        map_filename = f"{team}_special_teams_map.png"
    else:
        map_filename = f"{team}_relative.png"
        
    relative_map = f"league/{season}/{game_state}/{map_filename}"
    
    # We might want to pass some stats too, but for now just the map
    # We could load the summary.json to get stats for this team if needed
    
    return render_template("team_stats.html", 
                         season=season, 
                         game_state=game_state, 
                         team=team, 
                         relative_map=relative_map)

@app.route("/players")
def players():
    """Render the players statistics page."""
    import pandas as pd
    
    season = request.args.get('season', '20252026')
    
    # Paths
    # CSV: static/players/{season}/league/league_player_stats.csv
    # Scatter: static/players/{season}/league/league_scatter.png
    
    base_dir = os.path.join(ANALYSIS_DIR, "players", season, "league")
    csv_path = os.path.join(base_dir, "league_player_stats.csv")
    scatter_path = f"players/{season}/league/league_scatter.png"
    
    players_data = []
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Filter for min games?
            # df = df[df['games_played'] >= 5]
            
            # Convert to list of dicts
            # Round floats
            if 'xg_for_60' in df.columns:
                df['xg_for_60'] = df['xg_for_60'].round(2)
            if 'xg_against_60' in df.columns:
                df['xg_against_60'] = df['xg_against_60'].round(2)
            if 'toi_sec' in df.columns:
                df['toi_min'] = (df['toi_sec'] / 60).astype(int)
                
            # Handle new columns (GF, GA, CF, CA, etc.)
            # If they don't exist (yet), fill with None or 0
            new_cols = ['goals_for', 'goals_against', 'attempts_for', 'attempts_against', 'xgf_pct', 'gf_pct', 'cf_pct']
            for col in new_cols:
                if col not in df.columns:
                    df[col] = None # Or 0 if preferred, but None indicates missing data
                else:
                    # Round percentages
                    if 'pct' in col:
                         df[col] = df[col].round(1)
            
            players_data = df.to_dict('records')
            
        except Exception as e:
            logger.error(f"Failed to load player stats: {e}")
            
    return render_template("players.html", players=players_data, season=season, scatter_img=scatter_path)


@app.route("/player/<season>/<team>/<player_id>")
def player_view(season, team, player_id):
    """Render individual player view with relative map."""
    
    # Map path: static/players/{season}/{team}/{player_id}_relative.png
    # We pass the URL path to the template
    
    map_filename = f"{player_id}_relative.png"
    map_url = f"players/{season}/{team}/{map_filename}"
    
    # We could also load stats for this player if we want to display them
    # For now, just the map as requested
    
    return render_template("player_view.html", 
                         season=season, 
                         team=team, 
                         player_id=player_id, 
                         map_url=map_url)


@app.route("/monitor")
def monitor():
    """Render the log monitoring dashboard."""
    import time
    from datetime import datetime
    
    logs = []
    if os.path.exists(LOG_DIR):
        try:
            # List all files
            files = [f for f in os.listdir(LOG_DIR) if os.path.isfile(os.path.join(LOG_DIR, f))]
            # Sort by modification time (descending)
            files.sort(key=lambda x: os.path.getmtime(os.path.join(LOG_DIR, x)), reverse=True)
            
            for f in files:
                path = os.path.join(LOG_DIR, f)
                stat = os.stat(path)
                mtime = datetime.fromtimestamp(stat.st_mtime)
                size_bytes = stat.st_size
                
                # Format size
                if size_bytes < 1024:
                    size_str = f"{size_bytes} B"
                elif size_bytes < 1024 * 1024:
                    size_str = f"{size_bytes / 1024:.1f} KB"
                else:
                    size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
                
                # Determine status (crudely, by age)
                age = time.time() - stat.st_mtime
                if age < 60:
                    status = "Active"
                    status_class = "text-success" # Green
                elif age < 3600:
                    status = "Recent"
                    status_class = "text-warning" # Orange
                else:
                    status = "Old"
                    status_class = "text-muted" # Grey

                logs.append({
                    'name': f,
                    'mtime': mtime.strftime("%Y-%m-%d %H:%M:%S"),
                    'size': size_str,
                    'age_seconds': age,
                    'status': status,
                    'status_class': status_class
                })
        except Exception as e:
            logger.error(f"Failed to list logs: {e}")
            
    return render_template("monitor.html", logs=logs)


@app.route("/monitor/view/<filename>")
def monitor_view(filename):
    """Render the log view page."""
    
    path = os.path.join(LOG_DIR, filename)
    if not os.path.exists(path):
        return ("Log file not found", 404)
        
    try:
        # Check args
        tail_lines = request.args.get('tail')
        
        content = ""
        with open(path, 'r', errors='replace') as f:
            if tail_lines:
                try:
                    n = int(tail_lines)
                    # Simple tail implementation
                    # specific check for n=0 or small file omitted for brevity, 
                    # generic readlines is ok for reasonable sized logs.
                    # For huge logs, seek would be better.
                    lines = f.readlines()
                    content = "".join(lines[-n:])
                except ValueError:
                    content = f.read()
            else:
                 # Default to tail 500 if not specified to avoid crashing browser with massive logs
                 # unless 'full' is set
                 if request.args.get('full'):
                     content = f.read()
                 else:
                     lines = f.readlines()
                     content = "".join(lines[-1000:])
                     
    except Exception as e:
        return (f"Error reading log: {e}", 500)
        
    return render_template("log_view.html", filename=filename, content=content)


@app.route('/admin/flush_cache', methods=['POST'])
def admin_flush_cache():
    """Dev-only: clear the in-memory cache so static files are re-read.

    POST /admin/flush_cache
    """
    try:
        _CACHE.clear()
        logger.info('In-memory cache flushed via /admin/flush_cache')
        return ('cache flushed', 200)
    except Exception as e:
        logger.exception('failed to flush cache: %s', e)
        return (f'failed to flush cache: {e}', 500)


if __name__ == '__main__':
    logger.info('Starting Flask development server on http://0.0.0.0:5001')
    import sys
    sys.stdout.flush()
    # Disable reloader and debug mode to avoid "No space left on device" (ENOSPC/SemLock)
    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
