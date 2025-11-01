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
    """Endpoint to trigger re-plotting from the web UI.

    Accepts optional form field 'game' (integer). Calls plot_game.plot_shots synchronously
    and redirects back to the index page when finished.
    """
    # import here to avoid import-time side-effects if Flask isn't used
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
        # call the plotting routine (may fetch data) and save to static/shot_plot.png
        plot_game.plot_shots(game if game is not None else None, output_file=out_path)
    except Exception as e:
        return (f"Error generating plot: {e}", 500)

    # redirect back to the main page where the new plot will be visible
    return redirect(url_for('index'))


if __name__ == "__main__":
    # Run development server for local testing
    app.run(host="127.0.0.1", port=5000, debug=True)
