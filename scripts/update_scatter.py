import sys
import os
import json
import pandas as pd
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))

# Ensure matplotlib backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from puck.analyze import generate_player_scatter_plots

def run():
    season = '20252026'
    summary_path = f'analysis/players/{season}/player_summary_5v5.json'
    out_dir = f'analysis/players/{season}'

    if not os.path.exists(summary_path):
        print(f"Summary file not found: {summary_path}")
        return

    with open(summary_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} players from {summary_path}. Generating plots...")
    generate_player_scatter_plots(data, out_dir)
    print("Done.")

if __name__ == "__main__":
    run()
