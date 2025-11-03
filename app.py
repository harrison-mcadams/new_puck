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
import json
from pathlib import Path

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
    return render_template("index.html", exists=exists, img=img_filename)


@app.route("/replot", methods=["POST"])
def replot():
    """Trigger re-generation of the plot and redirect back to index.

    Accepts an optional form field `game` (integer). If omitted the code will
    attempt to use `plot_game.get_gameID(method='most_recent')` as a sensible
    default. Errors are returned as 500 responses for easier debugging.

    See module docstring for notes on improving this to a background job.
    """
    # Use the new pipeline: analyze the game to produce an events DataFrame
    # and render it using `plot.plot_events`.
    try:
        import plot
        import fit_xgs
    except Exception as e:
        return (f"plotting modules not available: {e}", 500)

    game = request.form.get('game') or None
    # attempt to coerce to int for downstream code that expects numeric gameIDs
    try:
        if game is not None and str(game).strip() != '':
            game_val = int(game)
        else:
            game_val = None
    except Exception:
        # keep as string if not an int (some feeds may accept string ids)
        game_val = game

    out_path = os.path.join(app.static_folder or 'static', 'shot_plot.png')

    try:
        # If no explicit game id, analyze_game will decide what to do.
        # Wrap the analysis + plotting in a try/except so network/HTTP errors
        # (e.g., NHL API access denied / 403 / 429) are handled gracefully.
        try:
            df = fit_xgs.analyze_game(game_val)
        except Exception as e:
            # Log the error and traceback for debugging, but continue by
            # creating a placeholder plot so the UI remains usable.
            import traceback, sys
            tb = traceback.format_exc()
            print('Error analyzing game (will produce placeholder plot):', e, file=sys.stderr)
            print(tb, file=sys.stderr)
            # create a placeholder plot (empty events) so the UI shows the rink
            try:
                plot.plot_events(pd.DataFrame([]), events_to_plot=['shot-on-goal', 'goal'], out_path=out_path)
            except Exception as e2:
                print('Error creating placeholder plot:', e2, file=sys.stderr)
            # Redirect back to index where the placeholder will be visible
            return redirect(url_for('index'))

        # If analysis succeeded but returned no rows, still make a placeholder
        if df is None or getattr(df, 'shape', (0,))[0] == 0:
            plot.plot_events(pd.DataFrame([]), events_to_plot=['shot-on-goal', 'goal'], out_path=out_path)
        else:
            # Call plot.plot_events to render the figure and save to out_path
            try:
                plot.plot_events(df, events_to_plot=['shot-on-goal','missed-shot','blocked-shot','goal','xgs'], out_path=out_path)
            except Exception as e:
                import traceback, sys
                tb = traceback.format_exc()
                print('Error plotting events; producing placeholder instead:', e, file=sys.stderr)
                print(tb, file=sys.stderr)
                try:
                    plot.plot_events(pd.DataFrame([]), events_to_plot=['shot-on-goal', 'goal'], out_path=out_path)
                except Exception as e2:
                    print('Error creating fallback placeholder plot:', e2, file=sys.stderr)
    except Exception as e:
        # Catch-all: log and return 500 with diagnostic information
        import traceback, sys
        tb = traceback.format_exc()
        print('Unhandled error in replot handler:', e, file=sys.stderr)
        print(tb, file=sys.stderr)
        return (f"Error generating plot: {e}", 500)

    # redirect back to the main page where the new plot will be visible
    return redirect(url_for('index'))


@app.route('/api/teams')
def api_teams():
    """Return JSON list of known teams for the season.

    Optional query param 'season' may be provided (defaults to '20252026').
    """
    from flask import jsonify, request
    season = request.args.get('season', '20252026')
    cache_key = f'teams:{season}'
    cached = _cache_get(cache_key)
    if cached is not None:
        return jsonify(cached)

    # First: try to serve from a static JSON file if present (no network required)
    static_teams_path = Path(app.static_folder or 'static') / 'teams.json'
    if static_teams_path.exists():
        try:
            with static_teams_path.open('r', encoding='utf-8') as fh:
                teams = json.load(fh)
                _cache_set(cache_key, teams)
                return jsonify(teams)
        except Exception as e:
            print('Failed to read static teams.json:', e, file=sys.stderr)

    # Allow forcing the default local list via environment for offline/dev
    if os.environ.get('USE_DEFAULT_TEAMS', '') == '1':
        return jsonify(DEFAULT_TEAMS)

    try:
        import nhl_api
        # get_season('all') pages weeks and returns game dicts containing team info
        games = nhl_api.get_season(team='all', season=season)
    except Exception as e:
        # Log the exception for debugging
        import sys
        print(f"Error fetching teams from NHL API: {e}", file=sys.stderr)
        # Fallback to default teams list
        return jsonify(DEFAULT_TEAMS)

    # Extract unique teams from games
    teams_map = {}
    for g in games or []:
        try:
            # defensive paths to find home/away teams in the game dict
            teams = g.get('teams') or {}
            if isinstance(teams, dict):
                for side in ('home','away'):
                    t = teams.get(side)
                    if isinstance(t, dict):
                        # t might be nested under 'team'
                        if 'team' in t and isinstance(t['team'], dict):
                            td = t['team']
                        else:
                            td = t
                        abb = td.get('triCode') or td.get('abbrev') or td.get('teamAbbrev') or td.get('abbreviation')
                        name = td.get('name') or td.get('teamName') or td.get('clubName')
                        if abb:
                            teams_map[str(abb).upper()] = {'abbr': str(abb).upper(), 'name': name or str(abb).upper()}
        except Exception:
            continue

    teams = sorted(list(teams_map.values()), key=lambda t: t['abbr'])
    _cache_set(cache_key, teams)
    return jsonify(teams)


@app.route('/api/games/<team>')
def api_team_games(team):
    """Return JSON list of games for the given team abbreviation.

    Optional query param 'season' may be provided (defaults to '20252026').
    """
    from flask import jsonify, request
    season = request.args.get('season', '20252026')
    team_abbr = (team or '').upper()
    cache_key = f'games:{team_abbr}:{season}'
    cached = _cache_get(cache_key)
    if cached is not None:
        return jsonify(cached)

    # Try static games file first
    static_games_path = Path(app.static_folder or 'static') / 'games_by_team.json'
    if static_games_path.exists():
        try:
            with static_games_path.open('r', encoding='utf-8') as fh:
                all_games = json.load(fh)
                team_games = all_games.get(team_abbr, [])
                _cache_set(cache_key, team_games)
                return jsonify(team_games)
        except Exception as e:
            import sys
            print('Failed to read static games_by_team.json:', e, file=sys.stderr)

    try:
        import nhl_api
        games = nhl_api.get_season(team=team_abbr, season=season)
    except Exception as e:
        # Log the error for diagnostics and return an empty list so the
        # frontend can still be used with manual Game ID entry.
        import sys
        print(f"Error fetching games from NHL API for {team_abbr}: {e}", file=sys.stderr)
        return jsonify([])

    out = []
    for g in games or []:
        try:
            gid = g.get('id') or g.get('gamePk') or g.get('gameID')
            if gid is None:
                continue
            # normalize id to int when possible
            try:
                gid = int(gid)
            except Exception:
                gid = str(gid)

            start = g.get('startTimeUTC') or g.get('gameDate') or g.get('startTime') or ''
            # derive opponent and home/away
            opp = ''
            homeaway = ''
            try:
                teams = g.get('teams') or {}
                if isinstance(teams, dict):
                    home = teams.get('home') or {}
                    away = teams.get('away') or {}
                    # extract inner 'team' dict if present
                    if isinstance(home, dict) and 'team' in home and isinstance(home['team'], dict):
                        home_team = home['team']
                    else:
                        home_team = home if isinstance(home, dict) else {}
                    if isinstance(away, dict) and 'team' in away and isinstance(away['team'], dict):
                        away_team = away['team']
                    else:
                        away_team = away if isinstance(away, dict) else {}

                    home_ab = (home_team.get('triCode') or home_team.get('abbrev') or home_team.get('teamAbbrev')) if isinstance(home_team, dict) else None
                    away_ab = (away_team.get('triCode') or away_team.get('abbrev') or away_team.get('teamAbbrev')) if isinstance(away_team, dict) else None
                    # Determine if requested team is home or away
                    if str(home_ab).upper() == team_abbr:
                        opp = away_ab or ''
                        homeaway = 'home'
                    elif str(away_ab).upper() == team_abbr:
                        opp = home_ab or ''
                        homeaway = 'away'
            except Exception:
                pass

            label = f"{start} - vs {opp} ({homeaway})" if opp else f"{start} - id {gid}"
            out.append({'id': gid, 'label': label, 'start': start})
        except Exception:
            continue

    # sort by start time where possible
    try:
        out_sorted = sorted(out, key=lambda x: x.get('start') or '')
    except Exception:
        out_sorted = out

    _cache_set(cache_key, out_sorted)
    return jsonify(out_sorted)


if __name__ == '__main__':
    # Run the Flask development server. This is intended for local development
    # only. In production use a WSGI server (gunicorn/uwsgi) and proper config.
    print('Starting Flask development server on http://127.0.0.1:5000')
    import sys
    sys.stdout.flush()
    app.run(host='127.0.0.1', port=5000, debug=True)
