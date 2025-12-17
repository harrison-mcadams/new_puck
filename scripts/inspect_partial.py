import numpy as np
import sys
import os

def inspect(path):
    print(f"Inspecting {path}...")
    try:
        data = np.load(path)
        print("Keys:", data.files)
        
        for key in data.files:
            if 'grid_team' in key:
                arr = data[key]
                print(f"Grid: {key}, Sum: {np.sum(arr):.4f}, Max: {np.max(arr):.4f}")
                
            elif 'stats' in key:
                # Stats is usually a JSON string or dict
                print(f"Stats: {key} (present)")
    except Exception as e:
        print(f"Error loading: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect(sys.argv[1])
    else:
        print("Usage: python inspect_partial.py <path>")
