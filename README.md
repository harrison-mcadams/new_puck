# Hockey Analytics Project

A comprehensive hockey analytics platform for analyzing NHL tracking data, calculating Expected Goals (xG), and visualizing team and player performance.

## Features

- **Expected Goals (xG) Model**: Random Forest model trained on distance, angle, and game state.
- **Rink Visualizations**: Shot maps, heatmaps, and relative performance maps.
- **Web Interface**: Flask-based web app for interactive analysis.
- **League Analysis**: Scripts to generate league-wide statistics and comparisons.
- **Player Analysis**: Detailed player-level metrics (xG rates, specific game states).

## Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Web Application

To start the web interface:

```bash
python app.py
```

Open your browser to `http://127.0.0.1:5000/`.

### Analysis Scripts

The `scripts/` directory contains various utilities for data processing and analysis.

-   **League Statistics**: Generate league-wide baselines and summaries.
    ```bash
    python scripts/run_league_stats.py
    ```

-   **Player Analysis**: Calculate stats and generate maps for individual players.
    ```bash
    python scripts/run_player_analysis.py
    ```

-   **Validation**: Validate xG model against external data (MoneyPuck).
    ```bash
    python scripts/validate_xgs.py
    ```

### Data Quality

Run quality checks on fetched data:

```bash
python scripts/qc_data.py
```

## Project Structure

-   `puck/`: Core analytics package (models, plotting, data parsing).
-   `web/`: Web application assets (templates, static files).
-   `scripts/`: Executable scripts for analysis and maintenance.
-   `data/`: Raw and processed data storage.
-   `static/`: Output directory for generated plots and reports.

## License

[License Information Here]
