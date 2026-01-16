
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck.impute import impute_blocked_shot_origins

def analyze_high_slot_concentration():
    print("--- Analyzing High Slot Blocked Shot Concentration ---")
    
    # 1. Define a REALISTIC distribution (Gaussian-ish centered at Net)
    # Most blocks happen deep!
    x_range = np.linspace(25, 85, 30)
    y_range = np.linspace(-40, 40, 20)
    X, Y = np.meshgrid(x_range, y_range)
    flat_x = X.flatten()
    flat_y = Y.flatten()
    
    # Calculate Weight driven by distance to Net (89, 0)
    dist = np.sqrt((flat_x - 89)**2 + flat_y**2)
    # Basic exponential decay freq ~ exp(-dist/20)
    weights = np.exp(-dist / 20.0)
    weights /= weights.sum()
    
    # Sample 10,000 shots based on this distribution
    N_SHOTS = 10000
    rng = np.random.default_rng(42)
    idxs = rng.choice(len(flat_x), size=N_SHOTS, p=weights)
    
    input_x = flat_x[idxs]
    input_y = flat_y[idxs]
    
    df = pd.DataFrame({
        'x': input_x,
        'y': input_y,
        'event': ['blocked-shot'] * N_SHOTS,
        'shooter_role': ['F'] * N_SHOTS,
        'home_team_defending_side': ['left'] * N_SHOTS
    })
    
    print(f"Simulating {len(df)} uniform blocked shots...")
    df_imp = impute_blocked_shot_origins(df, method='empirical_model', is_standardized=True)
    
    # 2. Analyze Density in High Slot (X=60 to 70)
    # Define zones
    def get_zone(x):
        if x < 45: return 'Point'
        if 45 <= x < 60: return 'High Zone'
        if 60 <= x < 75: return 'High Slot'
        if x >= 75: return 'Deep Slot'
        return 'Other'

    df['Input Zone'] = df['x'].apply(get_zone)
    df_imp['Output Zone'] = df_imp['imputed_x'].apply(get_zone)
    
    print("\n--- Migration Matrix (Where do blocks go?) ---")
    migration = pd.crosstab(df['Input Zone'], df_imp['Output Zone'], normalize='index') * 100
    print(migration.round(1))
    
    # 3. Check specific Deep Slot behavior
    deep_inputs = df[df['Input Zone'] == 'Deep Slot']
    deep_outputs = df_imp.loc[deep_inputs.index]
    
    pct_stayed_deep = (deep_outputs['Output Zone'] == 'Deep Slot').mean() * 100
    pct_moved_high = (deep_outputs['Output Zone'] == 'High Slot').mean() * 100
    
    print(f"\nDeep Slot Analysis:")
    print(f"  Inputs starting in Deep Slot: {len(deep_inputs)}")
    print(f"  % Remaining in Deep Slot: {pct_stayed_deep:.1f}%")
    print(f"  % Moved to High Slot: {pct_moved_high:.1f}%")

    # 4. Compare High Slot Volume
    high_slot_inputs = len(df[df['Input Zone'] == 'High Slot'])
    high_slot_outputs = len(df_imp[df_imp['Output Zone'] == 'High Slot'])
    ratio = high_slot_outputs / high_slot_inputs if high_slot_inputs > 0 else 0
    
    print(f"\nHigh Slot Concentration Factor:")
    print(f"  Uniform Inputs in High Slot: {high_slot_inputs}")
    print(f"  Imputed Outputs in High Slot: {high_slot_outputs}")
    print(f"  Multiplication Factor: {ratio:.2f}x")
    print(f"  (The High Slot has {ratio:.2f} times more blocked shots than expected from a uniform distribution)")

if __name__ == "__main__":
    analyze_high_slot_concentration()
