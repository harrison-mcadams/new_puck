import numpy as np
import os

# Path to a generated relative map
path = 'static/league/20252026/4v5/CBJ_relative_combined.npy'

if os.path.exists(path):
    data = np.load(path)
    print(f"Shape of relative map: {data.shape}")
else:
    print(f"File not found: {path}")
