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

app = Flask(__name__, static_folder="static", template_folder="templates")


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
        # If no explicit game id, attempt to use the same helper from fit_xgs
        if game_val is None:
            try:
                # fit_xgs.analyze_game expects a game id; fallback to None
                # and let analyze_game raise/use defaults. We cannot reliably
                # pick a most-recent game here without NHL API helper.
                game_val = None
            except Exception:
                game_val = None

        # Analyze game -> DataFrame of events
        df = fit_xgs.analyze_game(game_val)
        if df is None or df.shape[0] == 0:
            # still attempt to produce a placeholder plot via plot.plot_events
            plot.plot_events(pd.DataFrame([]), events_to_plot=['shot-on-goal','goal'], out_path=out_path)
        else:
            # Call plot.plot_events to render the figure and save to out_path
            plot.plot_events(df, events_to_plot=['shot-on-goal','missed-shot','blocked-shot','goal','xgs'], out_path=out_path)
    except Exception as e:
        return (f"Error generating plot: {e}", 500)

    # redirect back to the main page where the new plot will be visible
    return redirect(url_for('index'))


if __name__ == "__main__":
    # Run development server for local testing (do not use in production)
    app.run(host="127.0.0.1", port=5000, debug=True)
