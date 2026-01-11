
"""
Script to visualize Shot Density: Unblocked vs Blocked.
Split by Forward/Defenseman.
Shared colorbars for valid density comparison.
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import fit_xgs, data_pipeline
from puck.rink import draw_rink

def plot_hist_with_rink(ax, x, y, bins, vmin, vmax, cmap, title):
    """Plots a 2D histogram with rink background."""
    draw_rink(ax)
    
    # Filter to zone
    mask = (x >= 25) & (abs(y) <= 42.5)
    x = x[mask]
    y = y[mask]
    
    # Compute Hist
    h, xedges, yedges = np.histogram2d(x, y, bins=bins)
    
    # Plot using pcolormesh
    # Transpose H because histogram2d returns (nx, ny) but pcolormesh expects (ny, nx) logic usually or check x/y mapping
    # H[0,0] is at x[0], y[0]. 
    # meshgrid: X, Y
    X, Y = np.meshgrid(xedges, yedges)
    
    # H needs to be transposed for pcolormesh if X, Y are typically row/col ordered
    # correct is H.T
    mesh = ax.pcolormesh(X, Y, h.T, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.axis('off')
    return mesh

def main():
    print("Loading Data...")
    try:
        # Try explicit full load like train script
        df = fit_xgs.load_all_seasons_data()
    except:
        df = fit_xgs.load_data()
    print(f"Loaded {len(df)} rows.")

    print("Processing Pipeline (Imputation + Smearing)...")
    # Using the standard pipeline with Dithering enabled for smooth visualization
    df = data_pipeline.preprocess_features(
        df, 
        is_training=False, # We want to visualize the logic used for density, but maybe is_training=True enables dithering?
        apply_imputation=True,
        apply_dithering=True, # Enable dithering to verify smoothing effect
        apply_arena_adjustments=True
    )
    
    # Ensure is_blocked exists
    if 'is_blocked' not in df.columns:
        df['is_blocked'] = (df['event'] == 'blocked-shot').astype(int)
    
    df_f = df[df['shooter_role'] == 'F']
    df_d = df[df['shooter_role'] == 'D']
    
    print(f"Forwards: {len(df_f)}, Defensemen: {len(df_d)}")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Bins definition (approx 2ft square)
    x_bins = np.linspace(25, 89, 33) # ~2ft steps
    y_bins = np.linspace(-42.5, 42.5, 43)
    bins = [x_bins, y_bins]
    
    # --- Forwards ---
    print("Processing Forwards...")
    f_un = df_f[df_f['is_blocked'] == 0]
    f_bl = df_f[df_f['is_blocked'] == 1]
    
    # Compute max for shared scale
    h_f_un, _, _ = np.histogram2d(f_un['x'], f_un['y'], bins=bins)
    h_f_bl, _, _ = np.histogram2d(f_bl['x'], f_bl['y'], bins=bins)
    
    # Analyze distribution to handle extreme scaling (Forward presence in crease)
    all_counts_f = np.concatenate([h_f_un.flatten(), h_f_bl.flatten()])
    mask_pos = all_counts_f > 0
    if mask_pos.any():
        counts = all_counts_f[mask_pos]
        p50, p90, p95, p98, p99, p999 = np.percentile(counts, [50, 90, 95, 98, 99, 99.9])
        print(f"Forwards Counts: Max={counts.max()}")
        print(f"Percentiles: 50={p50:.0f}, 90={p90:.0f}, 95={p95:.0f}, 98={p98:.0f}, 99={p99:.0f}, 99.9={p999:.0f}")
        vmax_f = p98 # Aggressive clip to see the spread, not just the peak
    else:
        vmax_f = 1
        
    print(f"Forwards: Using Vmax={vmax_f}")
    
    # Plot
    m1 = plot_hist_with_rink(axes[0, 0], f_un['x'], f_un['y'], bins, 0, vmax_f, 'Blues', f"Forwards: Unblocked (N={len(f_un)})")
    m2 = plot_hist_with_rink(axes[0, 1], f_bl['x'], f_bl['y'], bins, 0, vmax_f, 'Reds', f"Forwards: Blocked (N={len(f_bl)})")
    
    # Colorbar for Forwards row
    cbar_ax_f = fig.add_axes([0.92, 0.55, 0.02, 0.35])
    fig.colorbar(m1, cax=cbar_ax_f, label='Shot Count (per ~2x2ft bin)')
    
    # --- Defensemen ---
    print("Processing Defensemen...")
    d_un = df_d[df_d['is_blocked'] == 0]
    d_bl = df_d[df_d['is_blocked'] == 1]
    
    h_d_un, _, _ = np.histogram2d(d_un['x'], d_un['y'], bins=bins)
    h_d_bl, _, _ = np.histogram2d(d_bl['x'], d_bl['y'], bins=bins)
    
    all_counts_d = np.concatenate([h_d_un.flatten(), h_d_bl.flatten()])
    mask_pos_d = all_counts_d > 0
    if mask_pos_d.any():
        counts_d = all_counts_d[mask_pos_d]
        p50, p90, p95, p98, p99 = np.percentile(counts_d, [50, 90, 95, 98, 99])
        print(f"Defensemen Counts: Max={counts_d.max()}")
        print(f"Percentiles: 50={p50:.0f}, 90={p90:.0f}, 95={p95:.0f}, 98={p98:.0f}, 99={p99:.0f}")
        vmax_d = p99 # Defensemen are less spiked, 99 is usually fine
    else:
        vmax_d = 1
        
    print(f"Defensemen: Using Vmax={vmax_d}")
    
    m3 = plot_hist_with_rink(axes[1, 0], d_un['x'], d_un['y'], bins, 0, vmax_d, 'Blues', f"Defensemen: Unblocked (N={len(d_un)})")
    m4 = plot_hist_with_rink(axes[1, 1], d_bl['x'], d_bl['y'], bins, 0, vmax_d, 'Reds', f"Defensemen: Blocked (N={len(d_bl)})")
    
    # Colorbar for Defensemen row (use m3 cmap/norm)
    cbar_ax_d = fig.add_axes([0.92, 0.1, 0.02, 0.35])
    fig.colorbar(m3, cax=cbar_ax_d, label='Shot Count (per ~2x2ft bin)')
    
    plt.subplots_adjust(right=0.9)
    out_path = 'analysis/shot_density_comparison.png'
    plt.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
