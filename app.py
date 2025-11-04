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

    # The frontend sets a single hidden input named 'game' (single source of truth).
    # Read only that field on the server to avoid mismatches.
    try:
        import sys
        print('replot: received form keys:', file=sys.stderr)
        for k in request.form.keys():
            print('  ', k, '=', request.form.get(k), file=sys.stderr)
    except Exception:
        pass

    raw_game = (request.form.get('game') or '').strip()
    if not raw_game:
        # Nothing provided — let analyze_game decide, but log the event.
        try:
            import sys
            print('replot: no game provided in form; analyze_game will choose default', file=sys.stderr)
        except Exception:
            pass
        game_val = None
    else:
        # normalize and validate
        cleaned = raw_game.replace(' ', '')
        if cleaned.isdigit():
            try:
                game_val = int(cleaned)
            except Exception:
                game_val = cleaned
        else:
            # allow non-digit ids as strings but strip whitespace
            game_val = cleaned

        try:
            import sys
            print(f"replot: received game field '{raw_game}' -> normalized '{game_val}'", file=sys.stderr)
        except Exception:
            pass

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


# Note: `/api/teams` and `/api/games/<team>` routes have been removed.
# The frontend now loads `static/teams.json` and `static/games_by_team.json`
# directly from the `/static` folder. This keeps the app simple and avoids
# runtime calls to the NHL API or any generation logic.


@app.route('/admin/flush_cache', methods=['POST'])
def admin_flush_cache():
    """Dev-only: clear the in-memory cache so static files are re-read.

    POST /admin/flush_cache
    """
    # In production you should protect this endpoint (auth/token). For local
    # development this is a convenience to avoid restarting the server.
    try:
        _CACHE.clear()
        print('In-memory cache flushed via /admin/flush_cache')
        return ('cache flushed', 200)
    except Exception as e:
        return (f'failed to flush cache: {e}', 500)


if __name__ == '__main__':
    # Run the Flask development server. This is intended for local development
    # only. In production use a WSGI server (gunicorn/uwsgi) and proper config.
    print('Starting Flask development server on http://127.0.0.1:5000')
    import sys
    sys.stdout.flush()
    app.run(host='127.0.0.1', port=5000, debug=True)
