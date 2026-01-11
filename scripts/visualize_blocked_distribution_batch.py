import sys
import os
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Force non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck.plot import draw_rink

def get_pbp_coords_from_json(game_id, block_ids):
    """
    Load game JSON and find x,y for given block_ids.
    Returns a dict {block_id: (x, y)}.
    """
    # Determine season from game_id (first 4 digits)
    # e.g. 2023020001 -> Season 20232024
    year_start = str(game_id)[:4]
    try:
        year_end = int(year_start) + 1
        season_str = f"{year_start}{year_end}"
    except ValueError:
        return {} # Invalid game_id

    path = os.path.join('data', season_str, f'game_{game_id}.json')
    if not os.path.exists(path):
        # Fallback: Try checking other likely folders if logic fails?
        # But this standard structure should work.
        # print(f"Warning: JSON not found: {path}", flush=True) 
        return {}
        
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        coords = {}
        target_ids = set(block_ids)
        
        for play in data.get('plays', []):
            eid = play.get('eventId') 
            if eid in target_ids:
                # Get coords
                details = play.get('details', {})
                x = details.get('xCoord')
                y = details.get('yCoord')
                if x is not None and y is not None:
                     coords[eid] = (x, y)
                     
        return coords
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return {}

def flip_to_one_side(x_arr, y_arr):
    """Reflect coordinates to positive X side."""
    x = np.array(x_arr)
    y = np.array(y_arr)
    
    # Flip
    mask = x < 0
    x[mask] = -x[mask]
    y[mask] = -y[mask]
    
    return x, y

def main():
    print("Starting Blocked Shot Visualization...", flush=True)

    # 1. Load Batch Data (Model Results)
    batch_path = os.path.join('analysis', 'blocked_shots', 'blocked_shots_summary_batch.csv')
    if not os.path.exists(batch_path):
        print(f"Error: {batch_path} not found.")
        return
    
    df_batch = pd.read_csv(batch_path)
    print(f"Loaded {len(df_batch)} batch records.", flush=True)

    # Filter Score > 0.4
    if 'score' in df_batch.columns:
        df_track = df_batch[df_batch['score'] > 0.4].copy()
    elif 'best_match_score' in df_batch.columns:
        df_track = df_batch[df_batch['best_match_score'] > 0.4].copy()
    else:
        df_track = df_batch.copy()
        
    print(f"Filtered (Score > 0.4): {len(df_track)} records.", flush=True)
    
    # 2. Lookup PBP Coordinates for these matched blocks
    print("Looking up PBP coordinates from JSON feeds...", flush=True)
    
    pbp_x_list = []
    pbp_y_list = []
    
    # We iterate by game to minimize file I/O
    # Add columns to df_track temporarily
    df_track['pbp_x'] = np.nan
    df_track['pbp_y'] = np.nan
    
    grouped = df_track.groupby('game_id')
    total_games = len(grouped)
    count = 0
    
    for game_id, group in grouped:
        count += 1
        if count % 50 == 0:
            print(f"Processing Game {count}/{total_games}...", flush=True)
            
        block_ids = group['block_id'].unique() # Assuming 'block_id' is the event ID
        coords_map = get_pbp_coords_from_json(game_id, block_ids)
        
        # Apply mapping
        for bid, (px, py) in coords_map.items():
            # Update rows
            mask = (df_track['game_id'] == game_id) & (df_track['block_id'] == bid)
            df_track.loc[mask, 'pbp_x'] = px
            df_track.loc[mask, 'pbp_y'] = py
            
    # Drop missing PBP matches
    df_matched = df_track.dropna(subset=['pbp_x', 'pbp_y'])
    print(f"Found PBP matches for {len(df_matched)} / {len(df_track)} records.", flush=True)
    
    # 3. Load All PBP Data (Background)
    pbp_csv_path = os.path.join('data', '20232024', '20232024_df.csv')
    print("Loading All PBP Data...", flush=True)
    # Just load minimal cols
    df_pbp_all = pd.read_csv(pbp_csv_path, usecols=['event', 'x', 'y'], low_memory=False)
    df_pbp_all = df_pbp_all[df_pbp_all['event'] == 'blocked-shot'].dropna(subset=['x', 'y'])
    print(f"Loaded {len(df_pbp_all)} total PBP blocked shots.", flush=True)

    # 4. Prepare Coordinates (Flip to one side)
    # Dataset 1: Edge (Tracking)
    edge_x, edge_y = flip_to_one_side(df_matched['x'].values, df_matched['y'].values) # 'x','y' in batch definition
    
    # Dataset 2: Matched PBP
    match_pbp_x, match_pbp_y = flip_to_one_side(df_matched['pbp_x'].values, df_matched['pbp_y'].values)
    
    # Dataset 3: All PBP
    all_pbp_x, all_pbp_y = flip_to_one_side(df_pbp_all['x'].values, df_pbp_all['y'].values)

    # 5. Plotting
    print("Generating Plots...", flush=True)
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1])
    
    def plot_hexbin(ax, x, y, title, cmap):
        draw_rink(ax)
        if len(x) > 0:
            try:
                ax.hexbin(x, y, gridsize=30, cmap=cmap, mincnt=1, alpha=0.9, extent=(0, 100, -42.5, 42.5))
            except Exception as e:
                print(f"Hexbin error: {e}")
                ax.scatter(x, y, alpha=0.1)
        ax.set_title(f"{title}\n(n={len(x)})")
        ax.set_xlim(0, 100)
        ax.set_ylim(-42.5, 42.5)
        ax.set_aspect('equal')

    # Row 1: Spatial
    ax1 = fig.add_subplot(gs[0, 0])
    plot_hexbin(ax1, edge_x, edge_y, "1) Block Locations from EDGE\n(Score > 0.4)", 'Blues')

    ax2 = fig.add_subplot(gs[0, 1])
    plot_hexbin(ax2, match_pbp_x, match_pbp_y, "2) Matched PBP Event Locations", 'Greens')

    ax3 = fig.add_subplot(gs[0, 2])
    plot_hexbin(ax3, all_pbp_x, all_pbp_y, "3) All PBP Blocked Shots\n(2023-24)", 'Reds')
    
    # Row 2: X-Histograms
    bins = np.linspace(0, 100, 50)
    
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(edge_x, bins=bins, color='blue', alpha=0.7, density=True)
    ax4.set_title("Edge X-Coord Distribution")
    ax4.set_xlabel("X Coordinate (0=Center, 100=End Board)")
    ax4.set_xlim(0, 100)

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(match_pbp_x, bins=bins, color='green', alpha=0.7, density=True)
    ax5.set_title("Matched PBP X-Coord Distribution")
    ax5.set_xlabel("X Coordinate")
    ax5.set_xlim(0, 100)

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(all_pbp_x, bins=bins, color='red', alpha=0.7, density=True)
    ax6.set_title("All PBP X-Coord Distribution")
    ax6.set_xlabel("X Coordinate")
    ax6.set_xlim(0, 100)

    plt.tight_layout()
    out_file = os.path.join('analysis', 'blocked_shot_distribution_batch.png')
    plt.savefig(out_file, dpi=150)
    print(f"Saved visualization to {out_file}", flush=True)

if __name__ == "__main__":
    main()
