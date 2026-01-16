
"""
Script to generate an interactive Shot Density Dashboard using Plotly.
Visualizes Unblocked vs Blocked shots for Forwards and Defensemen.
Values are binned counts with hover capability.
"""
import sys
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import fit_xgs, data_pipeline

def get_rink_shapes(xref='x', yref='y', x_offset=0):
    """
    Returns a list of Plotly layout shapes representing the rink.
    Focuses on the Offensive Zone (x >= 25).
    """
    shapes = []
    
    # Rink Dims
    # X: 25 to 100
    # Y: -42.5 to 42.5
    
    # Boards (Simple lines for now, ignoring rounded corners for performance/simplicity in dashboard)
    # Top Board
    shapes.append(dict(type="line", x0=25, y0=42.5, x1=100, y1=42.5, xref=xref, yref=yref, line=dict(color="black", width=2)))
    # Bottom Board
    shapes.append(dict(type="line", x0=25, y0=-42.5, x1=100, y1=-42.5, xref=xref, yref=yref, line=dict(color="black", width=2)))
    # End Board (approximate straight line for speed, real rink is curved)
    shapes.append(dict(type="line", x0=100, y0=-42.5, x1=100, y1=42.5, xref=xref, yref=yref, line=dict(color="black", width=2)))
    
    # Blue Line
    shapes.append(dict(type="line", x0=25, y0=-42.5, x1=25, y1=42.5, xref=xref, yref=yref, line=dict(color="blue", width=2)))
    
    # Goal Line (x=89)
    shapes.append(dict(type="line", x0=89, y0=-42.5, x1=89, y1=42.5, xref=xref, yref=yref, line=dict(color="red", width=1)))
    
    # Goal Crease (Simple semi-circle approximation or box)
    # Using a circle shape
    shapes.append(dict(type="circle", x0=89-4, y0=-4, x1=89+4, y1=4, xref=xref, yref=yref, line=dict(color="red", width=1)))
    
    # Faceoff Circles (69, 22) and (69, -22), R=15
    for y_center in [22, -22]:
        shapes.append(dict(type="circle", x0=69-15, y0=y_center-15, x1=69+15, y1=y_center+15, xref=xref, yref=yref, line=dict(color="red", width=1)))
        
    return shapes

def load_and_process_data():
    print("Loading Data...")
    try:
        df = fit_xgs.load_all_seasons_data()
    except:
        df = fit_xgs.load_data()
    print(f"Loaded {len(df)} rows.")

    print("Processing Pipeline...")
    df = data_pipeline.preprocess_features(
        df, 
        is_training=False,
        apply_imputation=True,
        apply_dithering=True,
        apply_arena_adjustments=True
    )
    
    if 'is_blocked' not in df.columns:
        df['is_blocked'] = (df['event'] == 'blocked-shot').astype(int)
        
    return df

def main():
    df = load_and_process_data()
    
    # Filter to O-Zone for plotting
    df = df[(df['x'] >= 25) & (df['y'].abs() <= 42.5)]
    
    # Filter for Valid Shot Events (Remove Faceoffs, Hits, etc.)
    valid_shots = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal']
    df = df[df['event'].isin(valid_shots)]
    
    # Split
    df_f = df[df['shooter_role'] == 'F']
    df_d = df[df['shooter_role'] == 'D']

    # -- Forwards Counts --
    f_un = df_f[df_f['is_blocked'] == 0]
    f_bl = df_f[df_f['is_blocked'] == 1]
    
    # -- Defensemen Counts --
    d_un = df_d[df_d['is_blocked'] == 0]
    d_bl = df_d[df_d['is_blocked'] == 1]
    
    # -- Defensemen Counts --
    d_un = df_d[df_d['is_blocked'] == 0]
    d_bl = df_d[df_d['is_blocked'] == 1]
    
    # --- DEBUG: Print Point Zone / High Slot Ratios ---
    print("\n--- DASHBOARD DATA DIAGNOSTIC ---")
    
    # Point Zone (X: 25-53, Y: +/-20)
    mask_point = (df['x'].between(25, 53)) & (df['y'].between(-20, 20))
    df_point = df[mask_point]
    n_point_bl = len(df_point[df_point['is_blocked'] == 1])
    n_point_all = len(df_point)
    ratio_point = n_point_bl / n_point_all if n_point_all > 0 else 0
    print(f"Point Zone (X 25-53): {n_point_bl}/{n_point_all} Blocked ({ratio_point:.1%})")
    
    # High Slot (X: 60-70, Y: +/-10)
    mask_high = (df['x'].between(60, 70)) & (df['y'].between(-10, 10))
    df_high = df[mask_high]
    n_high_bl = len(df_high[df_high['is_blocked'] == 1])
    n_high_all = len(df_high)
    ratio_high = n_high_bl / n_high_all if n_high_all > 0 else 0
    print(f"High Slot (X 60-70): {n_high_bl}/{n_high_all} Blocked ({ratio_high:.1%})")
    print("---------------------------------\n")

    # Subplots: 2 Rows (F, D), 3 Cols (Unblocked, Blocked, Difference)
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(f"Forwards: Unblocked (n={len(f_un)})", f"Forwards: Blocked (n={len(f_bl)})", "Forwards: Diff (Unblocked - Blocked)",
                        f"Defensemen: Unblocked (n={len(d_un)})", f"Defensemen: Blocked (n={len(d_bl)})", "Defensemen: Diff (Unblocked - Blocked)"),
        horizontal_spacing=0.03, vertical_spacing=0.1
    )
    
    # Define Bins (approx 2ft)
    nbinsx = int((100 - 25) / 2)
    nbinsy = int(85 / 2)
    
    # -- Forwards --
    # f_un and f_bl are already defined above
    
    # Calculate robust max for Forwards
    h_f_un, xedges, yedges = np.histogram2d(f_un['x'], f_un['y'], bins=[nbinsx, nbinsy], range=[[25, 100], [-42.5, 42.5]])
    h_f_bl, _, _ = np.histogram2d(f_bl['x'], f_bl['y'], bins=[nbinsx, nbinsy], range=[[25, 100], [-42.5, 42.5]])
    
    # Combined for Vmax
    h_f_all = np.concatenate([h_f_un.flatten(), h_f_bl.flatten()])
    vmax_f = np.percentile(h_f_all[h_f_all > 0], 98)
    print(f"Forwards Vmax (98th): {vmax_f}")
    
    # Difference (Unblocked - Blocked)
    h_f_diff = h_f_un - h_f_bl
    vmax_f_diff = max(abs(np.percentile(h_f_diff, 1)), abs(np.percentile(h_f_diff, 99))) # Robust symmetric max
    
    # Helper to create Heatmap from pre-computed histogram (Histogram2d doesn't do diffs easily)
    # We need to use Heatmap and provide midpoints of edges
    x_mids = (xedges[:-1] + xedges[1:]) / 2
    y_mids = (yedges[:-1] + yedges[1:]) / 2
    
    # Unblocked
    fig.add_trace(go.Heatmap(
        x=x_mids, y=y_mids, z=h_f_un.T,
        coloraxis="coloraxis1",
        hovertemplate="X: %{x}<br>Y: %{y}<br>Count: %{z}<extra></extra>"
    ), row=1, col=1)
    
    # Blocked
    fig.add_trace(go.Heatmap(
        x=x_mids, y=y_mids, z=h_f_bl.T,
        coloraxis="coloraxis1",
        hovertemplate="X: %{x}<br>Y: %{y}<br>Count: %{z}<extra></extra>"
    ), row=1, col=2)
    
    # Diff
    fig.add_trace(go.Heatmap(
        x=x_mids, y=y_mids, z=h_f_diff.T,
        coloraxis="coloraxis3",
        hovertemplate="X: %{x}<br>Y: %{y}<br>Diff: %{z}<extra></extra>"
    ), row=1, col=3)
    
    # -- Defensemen --
    # d_un and d_bl are already defined above
    
    h_d_un, _, _ = np.histogram2d(d_un['x'], d_un['y'], bins=[nbinsx, nbinsy], range=[[25, 100], [-42.5, 42.5]])
    h_d_bl, _, _ = np.histogram2d(d_bl['x'], d_bl['y'], bins=[nbinsx, nbinsy], range=[[25, 100], [-42.5, 42.5]])
    
    h_d_all = np.concatenate([h_d_un.flatten(), h_d_bl.flatten()])
    vmax_d = np.percentile(h_d_all[h_d_all > 0], 99)
    print(f"Defensemen Vmax (99th): {vmax_d}")
    
    h_d_diff = h_d_un - h_d_bl
    vmax_d_diff = max(abs(np.percentile(h_d_diff, 1)), abs(np.percentile(h_d_diff, 99)))

    # Unblocked
    fig.add_trace(go.Heatmap(
        x=x_mids, y=y_mids, z=h_d_un.T,
        coloraxis="coloraxis2",
        hovertemplate="X: %{x}<br>Y: %{y}<br>Count: %{z}<extra></extra>"
    ), row=2, col=1)
    
    # Blocked
    fig.add_trace(go.Heatmap(
        x=x_mids, y=y_mids, z=h_d_bl.T,
        coloraxis="coloraxis2",
        hovertemplate="X: %{x}<br>Y: %{y}<br>Count: %{z}<extra></extra>"
    ), row=2, col=2)
    
    # Diff
    fig.add_trace(go.Heatmap(
        x=x_mids, y=y_mids, z=h_d_diff.T,
        coloraxis="coloraxis4",
        hovertemplate="X: %{x}<br>Y: %{y}<br>Diff: %{z}<extra></extra>"
    ), row=2, col=3)
    
    # Layout Updates: Rink Shapes and Aspects
    full_shapes = []
    # 2 rows * 3 cols = 6 axes
    for i in range(1, 7):
        xref = 'x' if i == 1 else f'x{i}'
        yref = 'y' if i == 1 else f'y{i}'
        full_shapes.extend(get_rink_shapes(xref=xref, yref=yref))
    
    fig.update_layout(
        title="Shot Density Comparison: Unblocked vs Blocked vs Diff (Raw Counts)",
        shapes=full_shapes,
        width=1800, height=1000,
        coloraxis1=dict(colorscale='Blues', cmax=vmax_f, colorbar=dict(x=0.32, y=0.8, len=0.4, title="F Count")),
        coloraxis2=dict(colorscale='Blues', cmax=vmax_d, colorbar=dict(x=0.32, y=0.2, len=0.4, title="D Count")),
        coloraxis3=dict(colorscale='RdBu', cmin=-vmax_f_diff, cmax=vmax_f_diff, colorbar=dict(x=1.0, y=0.8, len=0.4, title="F Diff")),
        coloraxis4=dict(colorscale='RdBu', cmin=-vmax_d_diff, cmax=vmax_d_diff, colorbar=dict(x=1.0, y=0.2, len=0.4, title="D Diff")),
        showlegend=False
    )
    
    # Lock axes and aspect ratio
    # Set scaleanchor on Y-axis to lock to X-axis
    axes_names = ['yaxis'] + [f'yaxis{i}' for i in range(2, 7)]
    for i, axis in enumerate(axes_names):
        xref = 'x' if i == 0 else f'x{i+1}'
        fig.update_layout(**{axis: dict(range=[-42.5, 42.5], showgrid=False, scaleanchor=xref, scaleratio=1)})
        
    axes_names_x = ['xaxis'] + [f'xaxis{i}' for i in range(2, 7)]
    for axis in axes_names_x:
        fig.update_layout(**{axis: dict(range=[25, 100], showgrid=False)})

    out_path = 'analysis/shot_density_dashboard.html'
    fig.write_html(out_path)
    print(f"Dashboard saved to {out_path}")

if __name__ == "__main__":
    main()
