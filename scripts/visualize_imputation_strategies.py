
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck.impute import impute_blocked_shot_origins
from puck.plot import draw_rink

def plot_imputation_grid(ax, role='F', method='empirical_model', title=None):
    """
    Generates a grid of block locations in the defensive zone (0 to 100 x)
    and plots arrows pointing to their imputed origin.
    """
    draw_rink(ax)
    
    # Create Grid
    # Focused on defensive zone where blocks happen. 
    # Standard rink: 0-100 x, -42.5 to 42.5 y
    x_vals = np.linspace(25, 90, 15) # From blue line in?
    y_vals = np.linspace(-35, 35, 15)
    
    grid_x, grid_y = np.meshgrid(x_vals, y_vals)
    flat_x = grid_x.flatten()
    flat_y = grid_y.flatten()
    
    # Create DataFrame for API
    df_grid = pd.DataFrame({
        'x': flat_x,
        'y': flat_y,
        'event': 'blocked-shot',
        'shooter_role': role,
        'distance': np.sqrt((flat_x - 89)**2 + flat_y**2) # Approx distance to net
    })
    
    # Run Imputation
    # Note: Method='empirical_model' uses bins. 'cdf_mapping' uses quantile inversion.
    df_imputed = impute_blocked_shot_origins(df_grid, method=method, x_col='x', y_col='y', role_col='shooter_role')
    
    # Plot Vectors
    # Arrow from Block (x,y) -> Origin (imputed_x, imputed_y)
    # Actually, the imputation moves the origin BACKWARDS from the block?
    # No, we are imputing the Shot ORIGIN based on the Block Location.
    # So we want to see where the shot CAME FROM.
    # Arrow: Origin -> Block (Trajectory) OR Block -> Origin (Correction)?
    # Let's show Block -> Origin (Correction Vector) to see how we are adjusting.
    # Red dot = Block. Blue Arrow head = Origin.
    
    origin_x = df_imputed['imputed_x'].values.astype(float)
    origin_y = df_imputed['imputed_y'].values.astype(float)
    
    ax.scatter(flat_x, flat_y, c='red', s=10, alpha=0.5, label='Block Loc')
    
    # Vector: Origin - Block
    u = origin_x - flat_x
    v = origin_y - flat_y
    
    ax.quiver(flat_x, flat_y, u, v, angles='xy', scale_units='xy', scale=1, color='blue', width=0.003, alpha=0.8, headwidth=3)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Imputation: {method} ({role})")
        
    ax.set_xlim(0, 100)
    ax.set_ylim(-42.5, 42.5)
    ax.set_aspect('equal')

def plot_game_imputation(ax, game_id, method='empirical_model'):
    """
    Visualizes actual blocked shots from a specific game.
    """
    draw_rink(ax)
    
    # 1. Load Game JSON to get PBP
    # We need to find the file first (multi-season support)
    season_str = str(game_id)[:4] + str(int(str(game_id)[:4]) + 1)
    path = os.path.join('data', season_str, f'game_{game_id}.json')
    
    if not os.path.exists(path):
        print(f"Game file not found: {path}")
        return

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Extract blocks
    blocks = []
    for p in data.get('plays', []):
        if p.get('typeDescKey') == 'blocked-shot':
            details = p.get('details', {})
            x = details.get('xCoord')
            y = details.get('yCoord')
            if x is not None and y is not None:
                # We need shooter role? 
                # Currently we don't have easy role lookup without roster.
                # Default to 'F' or randomized for demo if logic allows.
                # In impute.py it defaults to 'F' if missing.
                blocks.append({'x': x, 'y': y, 'event': 'blocked-shot', 'shooter_role': 'F', 'distance': 0}) # Distance recalc inside
                
    if not blocks:
        ax.text(0.5, 0.5, "No Blocked Shots Found", transform=ax.transAxes)
        return
        
    df_blocks = pd.DataFrame(blocks)
    
    # Run Imputation
    df_imputed = impute_blocked_shot_origins(df_blocks, method=method)
    
    # Plot
    # We need to normalize visually to one side? Or keep full rink?
    # Keeping full rink is better for game context.
    
    for _, row in df_imputed.iterrows():
        bx, by = row['x'], row['y']
        ox, oy = row['imputed_x'], row['imputed_y']
        
        ax.plot([bx, ox], [by, oy], 'k-', alpha=0.3, linewidth=1)
        ax.plot(bx, by, 'rx', markersize=6, label='Block' if _ == 0 else "")
        ax.plot(ox, oy, 'bo', markersize=4, label='Imputed Origin' if _ == 0 else "")
        
    ax.set_title(f"Game {game_id}: Actual Blocked Shot Imputation\n(Method: {method})")
    ax.legend(loc='upper right')

def main():
    print("Generating Imputation Strategy Visualizations...", flush=True)
    
    fig = plt.figure(figsize=(20, 18))
    gs = fig.add_gridspec(3, 2)
    
    # 1. Grid: Empirical F
    ax1 = fig.add_subplot(gs[0, 0])
    plot_imputation_grid(ax1, role='F', method='empirical_model', title="Empirical Model (Forward)\nVector: Block -> Imputed Origin")
    
    # 2. Grid: Empirical D
    ax2 = fig.add_subplot(gs[0, 1])
    plot_imputation_grid(ax2, role='D', method='empirical_model', title="Empirical Model (Defender)\nVector: Block -> Imputed Origin")
    
    # 3. Grid: Quantile Mapping F
    ax3 = fig.add_subplot(gs[1, 0])
    plot_imputation_grid(ax3, role='F', method='quantile_matching', title="Quantile Mapping (Forward)\nVector: Block -> Imputed Origin")
    
    # 4. Grid: Quantile Mapping D
    ax4 = fig.add_subplot(gs[1, 1])
    plot_imputation_grid(ax4, role='D', method='quantile_matching', title="Quantile Mapping (Defender)\nVector: Block -> Imputed Origin")
    
    # 5. Game Viz (Bottom Row)
    ax_game = fig.add_subplot(gs[2, :])
    plot_game_imputation(ax_game, game_id=2025020670, method='empirical_model')
    
    plt.tight_layout()
    out_file = os.path.join('analysis', 'imputation_strategies_viz.png')
    os.makedirs('analysis', exist_ok=True)
    plt.savefig(out_file, dpi=150)
    print(f"Saved to {out_file}")

if __name__ == "__main__":
    main()
