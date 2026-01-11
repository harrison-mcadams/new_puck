
"""
Script to check for 'Default Coordinate' spikes at Faceoff Dots.
"""
import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from puck import fit_xgs

def main():
    print("Loading Data...")
    try:
        df = fit_xgs.load_all_seasons_data()
    except:
        df = fit_xgs.load_data()
        
    # Filter Forwards, Unblocked
    f_un = df[(df['shooter_role'] == 'F') & (df['event'] != 'blocked-shot')]
    
    # Faceoff Dots: (69, 22) and (69, -22)
    dots = [(69.0, 22.0), (69.0, -22.0)]
    
    print(f"Total Unblocked Forward Shots: {len(f_un)}")
    
    for x_dot, y_dot in dots:
        # Exact Match
        exact = f_un[(f_un['x'] == x_dot) & (f_un['y'] == y_dot)]
        print(f"\nExact matches at ({x_dot}, {y_dot}): {len(exact)}")
        
        # Nearby (within 1 ft)
        nearby = f_un[(f_un['x'].between(x_dot-1, x_dot+1)) & (f_un['y'].between(y_dot-1, y_dot+1))]
        print(f"Matches within +/- 1ft: {len(nearby)}")
        
        # Contrast with random point (e.g. 60, 22)
        random_x = 60.0
        exact_rand = f_un[(f_un['x'] == random_x) & (f_un['y'] == y_dot)]
        print(f"Exact matches at Control ({random_x}, {y_dot}): {len(exact_rand)}")

if __name__ == "__main__":
    main()
