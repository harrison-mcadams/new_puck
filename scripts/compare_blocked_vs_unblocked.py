import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def normalize_coordinates(df, x_col='x', y_col='y'):
    """
    Normalizes coordinates so all shots appear to be shooting towards the RIGHT net (Positive X).
    Standard NHL Rink: [-100, 100] X, [-42.5, 42.5] Y.
    If x < 0, we flip x and y.
    """
    df = df.copy()
    # Ensure numeric
    df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
    df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
    
    # Flip
    mask = df[x_col] < 0
    df.loc[mask, x_col] = -df.loc[mask, x_col]
    df.loc[mask, y_col] = -df.loc[mask, y_col]
    
    return df

def main():
    # Paths
    BLOCKED_SUMMARY = 'analysis/blocked_shots/blocked_shots_summary_batch.csv'
    PBP_DATA = 'data/20252026.csv'
    OUTPUT_DIR = 'analysis/blocked_shots'
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    print("Loading data...")
    if not os.path.exists(BLOCKED_SUMMARY):
        print(f"Error: {BLOCKED_SUMMARY} not found.")
        print("Please run the batch processor first or ensure the path is correct.")
        return

    # 1. Load Data
    try:
        df_pbp = pd.read_csv(PBP_DATA)
        print(f"Loaded {len(df_pbp)} PBP events.")
    except Exception as e:
        print(f"Failed to load PBP data: {e}")
        return
    
    # 2. Process Unblocked Shots (PBP)
    # Filter for intended shot events (Goals, Misses, ShotsMain)
    # Actual event names in CSV: 'shot-on-goal', 'goal', 'missed-shot'
    unblocked_events = ['shot-on-goal', 'goal', 'missed-shot']
    df_unblocked = df_pbp[df_pbp['event'].isin(unblocked_events)].copy()
    print(f"Filtered to {len(df_unblocked)} unblocked shot events from PBP.")
    
    # 3. Process Blocked Shots (PBP)
    # Use PBP data for blocks to ensure full sample (N~20k) vs unbiased tracking subset
    df_blocked = df_pbp[df_pbp['event'] == 'blocked-shot'].copy()
    print(f"Filtered to {len(df_blocked)} blocked shot events from PBP.")
    
    # Normalize Coordinates
    df_unblocked_norm = normalize_coordinates(df_unblocked, 'x', 'y')
    df_blocked_norm = normalize_coordinates(df_blocked, 'x', 'y')
    
    # 4. Filter for reasonable bounds (Half Ice)
    # Keep only Attacking Zone for cleaner plots (x > 25)
    df_unblocked_norm = df_unblocked_norm[df_unblocked_norm['x'] > 25]
    df_blocked_norm = df_blocked_norm[df_blocked_norm['x'] > 25]
    
    print(f"After spatial filtering (Attacking Zone):")
    print(f"  Unblocked: {len(df_unblocked_norm)}")
    print(f"  Blocked (PBP): {len(df_blocked_norm)}")
    
    # 5. Visualization
    # Setup styles
    sns.set_style("white")
    
    # --- PLOT 1: Side-by-Side KDE ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharex=True, sharey=True)
    
    # Common settings
    for ax in axes:
        ax.axvline(89, color='red', linestyle='-', alpha=0.3, label='Goal Line')
        ax.axvline(25, color='blue', linestyle='-', alpha=0.3, label='Blue Line')
        ax.set_xlim(25, 100)
        ax.set_ylim(-42.5, 42.5)
        ax.set_aspect('equal')
        ax.set_xlabel('X (ft)')

    # Left: Unblocked
    sns.kdeplot(
        data=df_unblocked_norm, x='x', y='y',
        fill=True, cmap="Blues", ax=axes[0], thresh=0.05, levels=15
    )
    axes[0].set_title(f'Unblocked Shots (N={len(df_unblocked_norm)})')
    axes[0].set_ylabel('Y (ft)')
    
    # Right: Blocked
    sns.kdeplot(
        data=df_blocked_norm, x='x', y='y',
        fill=True, cmap="Reds", ax=axes[1], thresh=0.05, levels=15
    )
    axes[1].set_title(f'Blocked Shots (N={len(df_blocked_norm)})')
    
    out_side = os.path.join(OUTPUT_DIR, 'blocked_vs_unblocked_side_by_side.png')
    plt.tight_layout()
    plt.savefig(out_side)
    print(f"Saved Side-by-Side plot to {out_side}")
    
    # --- PLOT 2: Contour Overlay (Unblocked Fill + Blocked Lines) ---
    plt.figure(figsize=(12, 10))
    plt.axvline(89, color='red', linestyle='-', alpha=0.3)
    plt.axvline(25, color='blue', linestyle='-', alpha=0.3)
    
    # Unblocked (Filled Blue)
    sns.kdeplot(
        data=df_unblocked_norm, x='x', y='y',
        fill=True, cmap="Blues", alpha=0.5, thresh=0.05, levels=10,
        label='Unblocked Distribution'
    )
    
    # Blocked (Red Contours - No Fill, Thicker Lines)
    sns.kdeplot(
        data=df_blocked_norm, x='x', y='y',
        fill=False, color="red", linewidths=2.0, thresh=0.05, levels=10,
        label='Blocked Distribution'
    )
    # Hack to create a custom legend handle since kdeplot(fill=False) doesn't always show up well in auto legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='blue', lw=4, alpha=0.5),
                    Line2D([0], [0], color='red', lw=2)]
    plt.legend(custom_lines, ['Unblocked (Fill)', 'Blocked (Contours)'], loc='upper left')
    
    plt.title('Comparison: Unblocked (Blue Area) vs Blocked (Red Contours)')
    plt.xlabel('X (ft)')
    plt.ylabel('Y (ft)')
    plt.xlim(25, 100)
    plt.ylim(-42.5, 42.5)
    plt.gca().set_aspect('equal')
    
    out_contour = os.path.join(OUTPUT_DIR, 'blocked_vs_unblocked_contours.png')
    plt.savefig(out_contour)
    print(f"Saved Contour plot to {out_contour}")


if __name__ == "__main__":
    main()
