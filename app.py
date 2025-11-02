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
    # import here to avoid import-time side-effects when importing this module
    # for tests or other tooling
    try:
        import plot_game
    except Exception as e:
        return (f"plotting module not available: {e}", 500)

    game = None
    try:
        game_val = request.form.get('game')
        if game_val:
            game = int(game_val)
    except Exception:
        game = None

    out_path = os.path.join(app.static_folder or 'static', 'shot_plot.png')
    try:
        # If no explicit game id, ask the plotting module for the most recent game
        if game is None:
            try:
                game = plot_game.get_gameID(method='most_recent')
            except Exception:
                game = None
        # Synchronous call that will block the request until finished.
        plot_game.plot_shots(game if game is not None else None, output_file=out_path)
    except Exception as e:
        return (f"Error generating plot: {e}", 500)

    # redirect back to the main page where the new plot will be visible
    return redirect(url_for('index'))


if __name__ == "__main__":
    # Run development server for local testing (do not use in production)
    app.run(host="127.0.0.1", port=5000, debug=True)
