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
    try:
        import plot
        import fit_xgs
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

    out_path = os.path.join(app.static_folder or 'static', 'shot_plot.png')

    try:
        try:
            df = fit_xgs.analyze_game(game_val)
        except Exception as e:
            # Log the error and traceback; produce a placeholder plot and continue
            logger.exception('Error analyzing game (will produce placeholder plot): %s', e)
            try:
                plot.plot_events(pd.DataFrame([]), events_to_plot=['shot-on-goal', 'goal'], out_path=out_path)
            except Exception as e2:
                logger.exception('Error creating placeholder plot: %s', e2)
            return redirect(url_for('index'))

        if df is None or getattr(df, 'shape', (0,))[0] == 0:
            plot.plot_events(pd.DataFrame([]), events_to_plot=['shot-on-goal', 'goal'], out_path=out_path)
        else:
            try:
                plot.plot_events(df, events_to_plot=['shot-on-goal','missed-shot','blocked-shot','goal','xgs'], out_path=out_path)
            except Exception as e:
                logger.exception('Error plotting events; producing placeholder instead: %s', e)
                try:
                    plot.plot_events(pd.DataFrame([]), events_to_plot=['shot-on-goal', 'goal'], out_path=out_path)
                except Exception as e2:
                    logger.exception('Error creating fallback placeholder plot: %s', e2)
    except Exception as e:
        logger.exception('Unhandled error in replot handler: %s', e)
        return (f"Error generating plot: {e}", 500)

    return redirect(url_for('index'))


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
    logger.info('Starting Flask development server on http://127.0.0.1:5000')
    import sys
    sys.stdout.flush()
    app.run(host='127.0.0.1', port=5000, debug=True)
