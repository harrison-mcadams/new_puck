
import json
import os
import pandas as pd
import numpy as np

def analyze_json():
    path = os.path.join("data", "arena_adjustments.json")
    if not os.path.exists(path):
        print("No adjustment file found.")
        return

    with open(path, 'r') as f:
        data = json.load(f)

    all_x = []
    all_y = []
    
    arena_stats = {}

    for season, s_data in data.items():
        if season == "20252026": continue # skip incomplete?
        
        for arena, a_data in s_data.items():
            x_vals = [abs(float(v)) for v in a_data.get('x', {}).values()]
            y_vals = [abs(float(v)) for v in a_data.get('y', {}).values()]
            
            all_x.extend(x_vals)
            all_y.extend(y_vals)
            
            mean_bias = np.mean(x_vals + y_vals) if (x_vals+y_vals) else 0
            if arena not in arena_stats:
                arena_stats[arena] = []
            arena_stats[arena].append(mean_bias)

    print("=== Theoretical Max Adjustments (from JSON) ===")
    print(f"X Adj: Mean={np.mean(all_x):.4f} ft, Max={np.max(all_x):.4f} ft")
    print(f"Y Adj: Mean={np.mean(all_y):.4f} ft, Max={np.max(all_y):.4f} ft")

    print("\n=== Top 5 Most Biased Arenas (Avg Adj Size) ===")
    # Avg across seasons
    avg_arena_stats = {k: np.mean(v) for k, v in arena_stats.items()}
    sorted_arenas = sorted(avg_arena_stats.items(), key=lambda x: x[1], reverse=True)
    for a, val in sorted_arenas[:5]:
        print(f"  {a}: {val:.4f} ft")

def analyze_csv():
    # Load a season that we know was backfilled
    csv_path = os.path.join("data", "20152016", "20152016_df.csv")
    if not os.path.exists(csv_path):
        print("\nCSV 20152016 not found (maybe backfill isn't done). Skipping CSV check.")
        return

    print(f"\n=== Empirical Adjustments (from {csv_path}) ===")
    df = pd.read_csv(csv_path)
    
    if 'x' not in df.columns or 'x_adj' not in df.columns:
        print("x or x_adj columns missing.")
        return
        
    # Calculate deltas
    df['dx'] = df['x_adj'] - df['x']
    df['dy'] = df['y_adj'] - df['y']
    df['dist'] = np.sqrt(df['dx']**2 + df['dy']**2)
    
    adjusted = df[df['dist'] > 0.001]
    
    print(f"Total Events: {len(df)}")
    print(f"Adjusted Events: {len(adjusted)} ({len(adjusted)/len(df)*100:.1f}%)")
    
    if not adjusted.empty:
        print(f"Mean Adjustment (Distance): {adjusted['dist'].mean():.4f} ft")
        print(f"Median Adjustment (Distance): {adjusted['dist'].median():.4f} ft")
        print(f"Max Adjustment (Distance): {adjusted['dist'].max():.4f} ft")
        print(f"95th Percentile: {adjusted['dist'].quantile(0.95):.4f} ft")
        
        # Check xG implications roughly?
        # Distance change?
        # Let's see if 1.5ft changes distance to goal significantly
        
        # Approx distance to center of goal (89, 0)
        # Simplified...
        df['r_old'] = np.sqrt((89 - df['x'].abs())**2 + df['y']**2)
        df['r_new'] = np.sqrt((89 - df['x_adj'].abs())**2 + df['y_adj']**2)
        df['r_diff'] = df['r_new'] - df['r_old']
        
        print(f"\nDistance to Goal Change (Delta R):")
        print(f"  Mean Abs Change: {df['r_diff'].abs().mean():.4f} ft")
        print(f"  Max Change: {df['r_diff'].abs().max():.4f} ft")

if __name__ == "__main__":
    analyze_json()
    analyze_csv()
