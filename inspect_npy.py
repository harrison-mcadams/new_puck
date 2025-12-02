import numpy as np
import os

path = 'static/league/20252026/5v5/ANA_relative_combined.npy'
if not os.path.exists(path):
    print(f"File not found: {path}")
else:
    data = np.load(path)
    print(f"Shape: {data.shape}")
    print(f"NaN count: {np.isnan(data).sum()}")
    
    valid_data = data[~np.isnan(data)]
    if valid_data.size > 0:
        print(f"Min: {valid_data.min()}")
        print(f"Max: {valid_data.max()}")
        print(f"Mean: {valid_data.mean()}")
        if np.all(valid_data == 0):
            print("Map is still entirely zeros.")
        else:
            print("Map has non-zero values!")
    else:
        print("Map is entirely NaNs.")
