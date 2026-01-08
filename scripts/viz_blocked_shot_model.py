
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import json
import os
import sys

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import rink

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../puck/data/blocked_shot_model.json')
OUT_DIR = os.path.join(os.path.dirname(__file__), '../analysis/blocked_shots_v2')

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        return None
    with open(MODEL_PATH, 'r') as f:
        return json.load(f)

def viz_vector_field(model):
    """
    Plots arrows from Block Location (Bin Center) to Mean Origin.
    """
    bins = model.get('bins', {})
    meta = model.get('meta', {})
    bin_size = meta.get('bin_size', 5.0)
    y_min = meta.get('y_min', -50.0)
    
    # Vectors
    X, Y, U, V, N = [], [], [], [], []
    
    for key, data in bins.items():
        k_x, k_y = map(int, key.split('_'))
        
        # Reconstruct Bin Center (Block Location)
        bx_center = k_x * bin_size + bin_size/2
        by_center = k_y * bin_size + y_min + bin_size/2
        
        mx = data['mx']
        my = data['my']
        n = data['n']
        
        # Filter sparse bins?
        if n < 3: continue
        
        X.append(bx_center)
        Y.append(by_center)
        
        # Vector: Tip - Tail
        U.append(mx - bx_center)
        V.append(my - by_center)
        
        N.append(n)
        
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    rink.draw_rink(ax)
    
    # Scale width/alpha by N?
    # Normalize N for color
    norm = mcolors.LogNorm(vmin=min(N), vmax=max(N))
    
    q = ax.quiver(X, Y, U, V, N, cmap='viridis', norm=norm, 
              angles='xy', scale_units='xy', scale=1, width=0.003, headwidth=4)
              
    plt.colorbar(q, label='Number of Observations')
    
    ax.set_title("Blocked Shot Imputation Vector Field\n(Arrow: Block Location -> Mean Origin)")
    
    # Add net marker
    ax.plot(89, 0, 'rx', markersize=10, markeredgewidth=3, label='Net')
    
    # Save
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
        
    out_path = os.path.join(OUT_DIR, 'imputation_vector_field.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.close()

def viz_origin_heatmap(model):
    """
    Scatter plot of Mean Origins, weighted/sized by N.
    Shows where "Ghost Shots" come from.
    """
    bins = model.get('bins', {})
    
    OX, OY, N, BX, BY = [], [], [], [], []
    
    for key, data in bins.items():
        if data['n'] < 3: continue
        OX.append(data['mx'])
        OY.append(data['my'])
        N.append(data['n'])
        
        # Reconstruct blocks for comparison context
        meta = model.get('meta', {})
        bin_size = meta.get('bin_size', 5.0)
        y_min = meta.get('y_min', -50.0)
        k_x, k_y = map(int, key.split('_'))
        BX.append(k_x * bin_size + bin_size/2)
        BY.append(k_y * bin_size + y_min + bin_size/2)
        
    fig, ax = plt.subplots(figsize=(10, 5))
    rink.draw_rink(ax)
    
    # Scatter Origins
    # Size by N
    sizes = np.array(N) * 2  # Scale factor
    
    sc = ax.scatter(OX, OY, s=sizes, c=OX, cmap='plasma', alpha=0.6, edgecolors='none', label='Imputed Origins')
    
    # Overlay Block Centers (small dots)
    ax.scatter(BX, BY, s=5, c='gray', alpha=0.3, label='Block Locations')
    
    ax.set_title("Distribution of Imputed Shot Origins (Sized by Frequency)")
    ax.legend()
    
    out_path = os.path.join(OUT_DIR, 'imputed_origins_scatter.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.close()

def viz_streamplot(model):
    """
    Streamplot to visualize the 'flow' from block to origin?
    Or maybe just interpolated field.
    """
    bins = model.get('bins', {})
    meta = model.get('meta', {})
    bin_size = meta.get('bin_size', 5.0)
    y_min = meta.get('y_min', -50.0)
    
    # Create grid
    xs = np.arange(0, 100, bin_size)
    ys = np.arange(-42.5, 42.5, bin_size)
    X, Y = np.meshgrid(xs, ys)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    
    # Fill grid
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            bx = X[i, j]
            by = Y[i, j]
            
            # Find key
            k_x = int(bx // bin_size)
            k_y = int((by - y_min) // bin_size)
            key = f"{k_x}_{k_y}"
            
            if key in bins:
                data = bins[key]
                # Direction vector
                dx = data['mx'] - bx
                dy = data['my'] - by
                U[i, j] = dx
                V[i, j] = dy
            else:
                U[i, j] = np.nan
                V[i, j] = np.nan
                
    # Plot 3: Pull Distance Heatmap
    fig, ax = plt.subplots(figsize=(10, 5))
    rink.draw_rink(ax)
    
    Z = np.sqrt(U**2 + V**2) # Magnitude of pull
    
    # Mask NaNs
    Z_masked = np.ma.masked_invalid(Z)
    
    mesh = ax.pcolormesh(X, Y, Z_masked, cmap='Reds', shading='auto', alpha=0.8)
    plt.colorbar(mesh, label='Imputation "Pull" Distance (ft)')
    ax.set_title("Magnitude of Coordinates Adjustment (Block -> Origin)")
    
    out_path = os.path.join(OUT_DIR, 'imputation_magnitude_heatmap.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.close()

def viz_sparsity_heatmap(model):
    """
    Heatmap of N (number of real observations) per bin.
    """
    bins = model.get('bins', {})
    meta = model.get('meta', {})
    bin_size = meta.get('bin_size', 5.0)
    y_min = meta.get('y_min', -50.0)
    
    # Create grid
    xs = np.arange(0, 100, bin_size)
    ys = np.arange(-50, 50, bin_size) # Note: Streamplot used -42.5 to 42.5
    
    # We want to cover the full bin range
    
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    
    # Fill grid
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            bx = X[i, j]
            by = Y[i, j]
            
            # Find key
            k_x = int(bx // bin_size)
            k_y = int((by - y_min) // bin_size)
            key = f"{k_x}_{k_y}"
            
            if key in bins:
                data = bins[key]
                Z[i, j] = data['n'] # Real observations
            else:
                Z[i, j] = 0
                
    fig, ax = plt.subplots(figsize=(10, 5))
    rink.draw_rink(ax)
    
    # Mask zeros? Maybe not, we want to see where it's 0.
    # But maybe mask the outside of rink areas?
    
    mesh = ax.pcolormesh(X, Y, Z, cmap='Blues', shading='auto', alpha=0.8, edgecolors='face')
    plt.colorbar(mesh, label='Real Observations (N)')
    ax.set_title("Model Sparsity: Observations per Bin")
    
    out_path = os.path.join(OUT_DIR, 'model_sparsity_heatmap.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.close()

def main():
    model = load_model()
    if not model: return
    
    print("Generating Vector Field...")
    viz_vector_field(model)
    
    print("Generating Origin Scatter...")
    viz_origin_heatmap(model)
    
    print("Generating Magnitude Heatmap...")
    viz_streamplot(model)
    
    print("Generating Sparsity Heatmap...")
    viz_sparsity_heatmap(model)

if __name__ == "__main__":
    main()
