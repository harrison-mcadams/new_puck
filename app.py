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

from flask import Flask, render_template, redirect, url_for, request
import os
import pandas as pd
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


app = Flask(__name__, static_folder="static", template_folder="templates")

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
    img_path = os.path.join(app.static_folder or "static", img_filename)
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
        import plot
        import fit_xgs
        import analyze
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

    out_path = os.path.join(app.static_folder or 'static', 'shot_plot.png')

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
            return_filtered_df=True
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
    summary_path = os.path.join(app.static_folder or "static", "league", season, game_state, f"{season}_team_summary.json")
    scatter_path = f"league/{season}/{game_state}/scatter.png"
    
    stats = []
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r') as f:
                stats = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load league stats: {e}")
            
    return render_template("league_stats.html", stats=stats, season=season, game_state=game_state, scatter_img=scatter_path)


@app.route("/team_maps")
def team_maps():
    """Render the gallery of team relative maps."""
    import glob
    
    # Get query parameters
    season = request.args.get('season', '20252026')
    game_state = request.args.get('game_state', '5v5')
    
    # Directory where maps are stored
    maps_dir = os.path.join(app.static_folder or "static", "league", season, game_state)
    
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
                    map_filename = f"{team}_relative_map.png"
                    map_path = os.path.join(maps_dir, map_filename)
                    
                    if os.path.exists(map_path):
                        teams_data.append({
                            'name': team,
                            'map_url': f"league/{season}/{game_state}/{map_filename}"
                        })
        except Exception as e:
            logger.error(f"Failed to load teams for maps: {e}")
    
    return render_template("team_maps.html", teams=teams_data, season=season, game_state=game_state)


@app.route("/team_stats/<season>/<game_state>/<team>")
def team_stats(season, game_state, team):
    """Render the team statistics page with relative map."""
    
    # Construct paths
    # Construct paths
    # Map is at static/league/{season}/{game_state}/{team}_relative_map.png
    relative_map = f"league/{season}/{game_state}/{team}_relative_map.png"
    
    # We might want to pass some stats too, but for now just the map
    # We could load the summary.json to get stats for this team if needed
    
    return render_template("team_stats.html", 
                         season=season, 
                         game_state=game_state, 
                         team=team, 
                         relative_map=relative_map)


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
    logger.info('Starting Flask development server on http://192.168.1.224:5000')
    import sys
    sys.stdout.flush()
    app.run(host='192.168.1.224', port=5000, debug=True)
