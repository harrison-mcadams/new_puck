
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

sys.path.append(os.getcwd())
from puck import analyze, features, rink

def main():
    print("--- Analyzing Crease Drop-off ---")
    model_path = 'analysis/xgs/xg_model_nested_all.joblib'
    clf = joblib.load(model_path)
    
    # Create a dense grid around the crease
    xs = np.linspace(80, 89, 20)  # 80ft to 89ft (Goal Line)
    ys = np.linspace(-5, 5, 20)   # +/- 5ft width
    X, Y = np.meshgrid(xs, ys)
    grid = pd.DataFrame({'x': X.ravel(), 'y': Y.ravel()})
    
    # Calculate Features
    goal_x = 89
    grid['distance'] = np.sqrt((grid['x'] - goal_x)**2 + grid['y']**2)
    
    # Angle Calc
    dx = grid['x'] - goal_x
    dy = grid['y']
    # Vector: (0, -1)
    # Cross = 0*dy - (-1)*dx = dx
    # Dot = 0*dx + (-1)*dy = -dy
    # Angle = arctan2(dx, -dy)
    # This logic seems consistent with previous script but let's verify standard calc
    # Angle in degrees from center line logic? 
    # Let's use simple logic: angle=0 is straight on? No, angle usually 0 at goal line?
    # Let's use the exact logic from analyze_model_spatial_weights.py to be consistent
    rx, ry = 0.0, -1.0
    cross = rx * dy - ry * dx
    dot = rx * dx + ry * dy
    angle_rad = np.arctan2(cross, dot)
    grid['angle_deg'] = (-np.degrees(angle_rad)) % 360.0

    # Constants
    grid['game_state'] = '5v5'
    grid['shot_type'] = 'wrist'
    grid['is_rebound'] = 0
    grid['rebound_angle_change'] = 0
    grid['rebound_time_diff'] = 0
    grid['is_rush'] = 0
    grid['last_event_type'] = 'facetoff'
    grid['last_event_time_diff'] = 10
    grid['shooter_role'] = 'F'
    grid['shoots_catches'] = 'L'
    grid['score_diff'] = 0
    grid['period_number'] = 1
    grid['time_elapsed_in_period_s'] = 600
    grid['total_time_elapsed_s'] = 600
    grid['event'] = 'shot-on-goal'
    
    # Predict
    grid['xg'] = clf.predict_proba(grid)[:, 1]
    
    # Print a cross-section at y=-1.5 (where we saw the drop)
    print("\nCross-section at y ~ -1.5:")
    subset = grid[np.isclose(grid['y'], -1.84, atol=0.5)].sort_values('x')
    print(subset[['x', 'y', 'distance', 'angle_deg', 'xg']])

    # Plot
    plt.figure(figsize=(10, 8))
    plt.tricontourf(grid['x'], grid['y'], grid['xg'], levels=20, cmap='viridis')
    plt.colorbar(label='Predicted xG')
    plt.title("Granular xG in Crease (80-89ft)")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.axvline(89, color='red', linestyle='--', label='Goal Line')
    plt.legend()
    plt.savefig('analysis/nested_xgs/crease_deepdive.png')
    print("\nPlot saved to analysis/nested_xgs/crease_deepdive.png")

if __name__ == "__main__":
    main()
