
import numpy as np
import os

def check():
    path = "analysis/league/20252026/5v5/baseline.npy"
    if not os.path.exists(path):
        print("Baseline not found")
        return
        
    base = np.load(path)
    print(f"Baseline shape: {base.shape}")
    print(f"Min: {np.nanmin(base)}")
    print(f"Max: {np.nanmax(base)}")
    print(f"Mean: {np.nanmean(base)}")
    print(f"Has NaNs: {np.isnan(base).any()}")
    
if __name__ == "__main__":
    check()
