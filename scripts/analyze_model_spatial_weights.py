
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())
from puck import analyze, features, rink

def main():
    print("--- Nested xG Spatial Weight Deep-Dive ---")
    
    # 1. Load Model
    model_path = 'analysis/xgs/xg_model_nested_all.joblib'
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Wait for training to finish.")
        return
    
    clf = joblib.load(model_path)
    print(f"Loaded {type(clf).__name__}")

    # 2. Create Evaluation Grid
    xs = np.linspace(0, 100, 101)
    ys = np.linspace(-42.5, 42.5, 86)
    X, Y = np.meshgrid(xs, ys)
    grid_points = pd.DataFrame({'x': X.ravel(), 'y': Y.ravel()})
    
    # Add baseline features
    goal_x, goal_y = 89, 0
    dx = grid_points['x'] - goal_x
    dy = grid_points['y'] - goal_y
    grid_points['distance'] = np.sqrt(dx**2 + dy**2)
    
    # Vectorized angle calculation
    # Reference vector along goal line pointing toward goalie's left (for right goal, goalie faces -x, left is -y)
    rx, ry = 0.0, -1.0
    cross = rx * dy - ry * dx
    dot = rx * dx + ry * dy
    angle_rad_ccw = np.arctan2(cross, dot)
    grid_points['angle_deg'] = (-np.degrees(angle_rad_ccw)) % 360.0
    
    grid_points['game_state'] = '5v5'
    grid_points['shot_type'] = 'wrist'
    grid_points['is_rebound'] = 0
    grid_points['rebound_angle_change'] = 0
    grid_points['rebound_time_diff'] = 0
    grid_points['is_rush'] = 0
    grid_points['last_event_type'] = 'facetoff'
    grid_points['last_event_time_diff'] = 10
    grid_points['shooter_role'] = 'F'
    grid_points['shoots_catches'] = 'L'
    grid_points['score_diff'] = 0
    grid_points['period_number'] = 1
    grid_points['time_elapsed_in_period_s'] = 600
    grid_points['total_time_elapsed_s'] = 600
    grid_points['event'] = 'shot-on-goal' # To avoid 0.0 override

    # 3. Analyze "Shot Type" Influence (Slap vs Wrist)
    print("Analyzing Shot Type Delta...")
    
    # DEBUG: Check Crease Values specifically
    crease_mask = (grid_points['x'] >= 85) & (grid_points['x'] <= 89) & (grid_points['y'].abs() <= 2)
    print(grid_points[crease_mask][['x', 'y', 'distance', 'angle_deg']].head())
    
    grid_wrist = grid_points.copy()
    grid_slap = grid_points.copy()
    grid_slap['shot_type'] = 'slap'
    
    # Use predict_proba directly to get xG
    # Note: predict_proba handles categorical encoding internally via preprocess_data
    xg_wrist = clf.predict_proba(grid_wrist)[:, 1]
    
    # DEBUG: Check probabilities in crease
    grid_wrist['prob'] = xg_wrist
    print("\n[DEBUG] Predicted Probs in Crease (Wrist, 5v5):")
    print(grid_wrist[crease_mask][['x', 'y', 'prob']].head())

    xg_slap = clf.predict_proba(grid_slap)[:, 1]
    
    delta_slap = xg_slap - xg_wrist
    
    # 4. Analyze "Rebound" Influence
    print("Analyzing Rebound Delta...")
    grid_rebound = grid_points.copy()
    grid_rebound['is_rebound'] = 1
    grid_rebound['rebound_angle_change'] = 45
    grid_rebound['rebound_time_diff'] = 1.0
    grid_rebound['last_event_time_diff'] = 1.0
    
    xg_rebound = clf.predict_proba(grid_rebound)[:, 1]
    delta_rebound = xg_rebound - xg_wrist
    
    # 5. Plotting
    # 5. Plotting
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    def plot_heatmap(vals, title, ax):
        # Draw Rink Background
        rink.draw_rink(ax, show_goals=True)
        
        # Plot Data Contour
        # Use tricontourf for smooth filling over the rink schematic
        # Clip to offensive zone for better detail
        ax.set_xlim(25, 100)
        ax.set_ylim(-42.5, 42.5)
        
        # Create a triangulation? Or just use tricontourf directly on x,y
        c = ax.tricontourf(grid_points['x'], grid_points['y'], vals, levels=20, cmap='viridis', alpha=0.8)
        fig.colorbar(c, ax=ax)
        ax.set_title(title)

    plot_heatmap(xg_wrist, "Baseline xG (5v5 Wrist Shot)", axes[0, 0])
    plot_heatmap(delta_slap, "Delta xG: Slap Shot vs Wrist Shot", axes[0, 1])
    plot_heatmap(delta_rebound, "Delta xG: Rebound vs Baseline", axes[1, 0])
    
    # Feature Interaction: Distance Gradient
    # Just show xG at D role vs F role
    grid_d = grid_points.copy()
    grid_d['shooter_role'] = 'D'
    xg_d = clf.predict_proba(grid_d)[:, 1]
    plot_heatmap(xg_d - xg_wrist, "Delta xG: Defenseman vs Forward", axes[1, 1])

    plt.tight_layout()
    out_path = 'analysis/nested_xgs/spatial_feature_deepdive.png'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    print(f"Deep-dive plots saved to {out_path}")

if __name__ == "__main__":
    main()
