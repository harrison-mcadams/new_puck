
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck.impute import impute_blocked_shot_origins

def visualize_flow():
    print("--- Generating Imputation Flow Map ---")
    
    # Define Grid (Standardized Attack Right)
    x_range = np.linspace(25, 85, 12)  # X from Blue Line to near Goal
    y_range = np.linspace(-40, 40, 9)  # Y across width
    X, Y = np.meshgrid(x_range, y_range)
    
    # Flatten for processing
    flat_x = X.flatten()
    flat_y = Y.flatten()
    N = len(flat_x)
    
    # Create Df
    # Run multiple samples per point to get a smooth average vector
    SAMPLES_PER_POINT = 20
    
    input_x = np.repeat(flat_x, SAMPLES_PER_POINT)
    input_y = np.repeat(flat_y, SAMPLES_PER_POINT)
    
    df = pd.DataFrame({
        'x': input_x,
        'y': input_y,
        'event': ['blocked-shot'] * len(input_x),
        'shooter_role': ['F'] * len(input_x),
        'home_team_defending_side': ['left'] * len(input_x)
    })
    
    print(f"Simulating {len(df)} blocked shots...")
    df_imp = impute_blocked_shot_origins(df, method='empirical_model', is_standardized=True)
    
    # Group back by original grid point to get average Imputed X/Y
    df_imp['grid_x'] = input_x
    df_imp['grid_y'] = input_y
    
    grouped = df_imp.groupby(['grid_x', 'grid_y'])[['imputed_x', 'imputed_y']].mean().reset_index()
    
    # Setup Vectors
    # U = dx (Imputed X - Block X)
    # V = dy (Imputed Y - Block Y)
    U = grouped['imputed_x'] - grouped['grid_x']
    V = grouped['imputed_y'] - grouped['grid_y']
    
    # Calc Physics Violation for coloring
    # Violation if Shooter is closer to Net (89, 0) than Block
    d_block = np.sqrt((grouped['grid_x'] - 89)**2 + grouped['grid_y']**2)
    d_shoot = np.sqrt((grouped['imputed_x'] - 89)**2 + grouped['imputed_y']**2)
    violations = d_shoot < d_block
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Rink Outline (roughly)
    plt.axvline(25, color='blue', alpha=0.3, linestyle='--', label='Blue Line')
    plt.plot(89, 0, 'ro', markersize=10, label='Net')
    plt.xlim(0, 100)
    plt.ylim(-45, 45)
    
    # Quiver
    # Color condition: Red if violation (on average), Green/Blue if good
    # We sadly can't easily color individual arrows in basic quiver without some work, 
    # but we can do two quiver calls.
    
    # 1. Valid Flows
    mask_valid = ~violations
    plt.quiver(grouped.loc[mask_valid, 'grid_x'], grouped.loc[mask_valid, 'grid_y'], 
               U[mask_valid], V[mask_valid], 
               color='green', alpha=0.6, scale=1, scale_units='xy', angles='xy', label='Valid (Further)')
               
    # 2. Invalid Flows
    mask_invalid = violations
    plt.quiver(grouped.loc[mask_invalid, 'grid_x'], grouped.loc[mask_invalid, 'grid_y'], 
               U[mask_invalid], V[mask_invalid], 
               color='red', alpha=0.8, scale=1, scale_units='xy', angles='xy', label='Violation (Closer)')
    
    plt.title('Imputation Flow Map: Average Displacement (Block -> Shooter)')
    plt.xlabel('Standardized X (Goal at 89)')
    plt.ylabel('Standardized Y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = 'analysis/imputation_flow.png'
    plt.savefig(out_path)
    print(f"Saved flow map to {out_path}")
    
    # Text Summary
    print("\nFlow Summary (Center Lane X, Y=0):")
    center_mask = (grouped['grid_y'] == 0.0)
    center_data = grouped[center_mask].sort_values('grid_x')
    for _, row in center_data.iterrows():
        status = "VIOLATION" if (np.abs(row['imputed_x'] - 89) < np.abs(row['grid_x'] - 89)) else "OK"
        print(f"Block X={row['grid_x']:.0f} -> Imputed X={row['imputed_x']:.1f} ({status})")

if __name__ == "__main__":
    visualize_flow()
