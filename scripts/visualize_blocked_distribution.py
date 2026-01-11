import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck.plot import draw_rink

def main():
    # 1. Load Data
    print("Loading data...")
    
    # Tracking Comparisons (Empirical)
    track_path = os.path.join('analysis', 'blocked_shot_comparison.csv')
    if not os.path.exists(track_path):
        print(f"Error: {track_path} not found. Run compare_block_locations.py first.")
        return
    df_track = pd.read_csv(track_path)
    
    # PBP Data (All Blocked Shots)
    pbp_path = os.path.join('data', '20242025', '20242025_df.csv')
    if not os.path.exists(pbp_path):
         print(f"Error: {pbp_path} not found.")
         return
    
    # Load PBP columns of interest only to save memory if needed, but it's small enough.
    df_pbp = pd.read_csv(pbp_path, low_memory=False)
    df_pbp_blocks = df_pbp[df_pbp['event'] == 'blocked-shot'].copy()
    
    print(f"Loaded {len(df_track)} empirical blocks and {len(df_pbp_blocks)} PBP blocks.")
    
    # 2. Setup Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Empirical (Tracking)
    ax1 = axes[0]
    draw_rink(ax1)
    ax1.set_title(f"Empirical Tracking Block Locations (n={len(df_track)})")
    # Tracking coords are already normalized [-100, 100] in previous script
    # Plot them.
    # Color by error? Or just locations. User asked for "shot locations".
    # Let's use blue for empirical.
    ax1.scatter(df_track['track_x'], df_track['track_y'], 
                alpha=0.6, s=20, c='blue', edgecolors='none', label='Tracking Impact')
    # Maybe overlay PBP for comparison? User asked for separate subplots relative to "our empiric" vs "pbp blocked shots".
    # User said: "show me the shot locations of our empiric... on another subplot... show me the same for the pbp blocked shots"
    
    # Plot 2: PBP Blocked Shots
    ax2 = axes[1]
    draw_rink(ax2)
    ax2.set_title(f"All PBP Blocked Shot Locations (2024-25, n={len(df_pbp_blocks)})")
    
    # Need to handle PBP coordinates?
    # PBP x,y are usually raw.
    # In puck/plot.py logic it adjusts for home/away.
    # For a raw distribution, we might want to see them "as recorded".
    # Usually they are recorded relative to the attacking goal?
    # Let's just plot 'x' and 'y'. If they are messed up, the plot will show it (density in weird spots).
    # But usually PBP x,y needs standardizing if we want them all on one side.
    # "Visualize the spatial distribution" -> implies we want to see where blocks happen on the ice.
    # If we don't normalize direction, they will be on both ends.
    # Empirical ones from my script: `identify_shot_attempts` logic didn't explicitly flip for possession.
    # Wait, `identify_shot_attempts` just normalized x to [-100, 100]. It didn't flip for "Attacking Right".
    # So empirical shots will be on both sides unless I flip them.
    # To compare distributions effectively, we should probably FLIP everything to one zone (e.g. Attacking Right).
    
    # Helper to flip to right half (x > 0)
    # Actually, blocks happen in DEFENSIVE zone usually.
    # So we should flip to Defensive Zone (e.g. x < 0) or just one side.
    # Let's flip all to Positive X (0 to 100) for cleaner "Distribution" view.
    
    def flip_to_one_side(x, y):
        # If play is symmetrical, just abs(x)?
        # No, y needs to flip if x flips to maintain handedness?
        # Usually: x_new = abs(x). y_new = y * sign(x)? (-x, -y) is 180 rotation.
        # Simple reflection: x_new = abs(x). y_new = y. (Mirror).
        # But rotation is better.
        # If x < 0: x = -x, y = -y.
        new_x = x.copy()
        new_y = y.copy()
        mask = new_x < 0
        new_x[mask] = -new_x[mask]
        new_y[mask] = -new_y[mask]
        return new_x, new_y

    # Flip Tracking
    tx, ty = flip_to_one_side(df_track['track_x'].values, df_track['track_y'].values)
    ax1.clear()
    draw_rink(ax1)
    ax1.set_title(f"Empirical Tracking Blocks (Normalized to One Zone)\nn={len(df_track)}")
    ax1.scatter(tx, ty, alpha=0.5, s=20, c='blue', label='Tracking')
    
    # Flip PBP
    # Filter out NaNs
    df_pbp_blocks = df_pbp_blocks.dropna(subset=['x', 'y'])
    px, py = flip_to_one_side(df_pbp_blocks['x'].values, df_pbp_blocks['y'].values)
    ax2.clear()
    draw_rink(ax2)
    ax2.set_title(f"All PBP Blocks (Normalized to One Zone)\nn={len(df_pbp_blocks)}")
    ax2.scatter(px, py, alpha=0.1, s=10, c='red', label='PBP') # High alpha for density
    
    # Save
    out_file = os.path.join('analysis', 'blocked_shot_distribution.png')
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {out_file}")

if __name__ == "__main__":
    main()
