import json
import os
import sys
import matplotlib
matplotlib.use('Agg') # Prevent GUI hang
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("Debug: Project root added to path")

# Standalone Rink Drawer to avoid import hangs
class RinkDrawer:
    def draw_rink(self, ax=None):
        if ax is None: fig, ax = plt.subplots(figsize=(10, 5))
        # Draw Boards
        rect = patches.Rectangle((-100, -42.5), 200, 85, linewidth=2, edgecolor='black', facecolor='white', zorder=0)
        ax.add_patch(rect)
        # Center Line
        ax.plot([0, 0], [-42.5, 42.5], color='red', linewidth=2)
        # Blue Lines
        ax.plot([25, 25], [-42.5, 42.5], color='blue', linewidth=2)
        ax.plot([-25, -25], [-42.5, 42.5], color='blue', linewidth=2)
        # Goal Lines
        ax.plot([89, 89], [-42.5, 42.5], color='red', linewidth=1)
        ax.plot([-89, -89], [-42.5, 42.5], color='red', linewidth=1)
        
        # Faceoff Dots
        ax.scatter([20, 69, -20, -69], [22, 22, 22, 22], c='red', s=20)
        ax.scatter([20, 69, -20, -69], [-22, -22, -22, -22], c='red', s=20)
        
        ax.set_xlim(-100, 100)
        ax.set_ylim(-42.5, 42.5)
        ax.set_aspect('equal')
        return ax

rink = RinkDrawer()

def plot_model_vectors(model_path, title, out_path):
    print(f"Debug: Starting processing for {os.path.basename(model_path)}")
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    print(f"Loading {model_path}...")
    with open(model_path, 'r') as f:
        data = json.load(f)
        
    bins = data.get('bins', {})
    meta = data.get('meta', {})
    bin_size = meta.get('bin_size', 5.0)
    y_min = meta.get('y_min', -50.0)
    
    # Setup Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    rink.draw_rink(ax)
    
    # Focus on Attacking Zone (Right Side, X > 0)
    ax.set_xlim(0, 100)
    ax.set_ylim(-42.5, 42.5)
    
    count = 0
    
    # Colors for magnitude of correction?
    # Or just standard vectors
    
    for key, info in bins.items():
        try:
            kx, ky = map(int, key.split('_'))
            
            # Bin Center (Block Location)
            bx = kx * bin_size + (bin_size / 2)
            by = (ky * bin_size) + y_min + (bin_size / 2)
            
            # Imputed Origin
            mx = info['mx']
            my = info['my']
            
            n = info.get('n', 1)
            
            # Filter low sample size?
            if n < 3: continue
            
            # Plot Vector
            # Arrow from Block (bx, by) to Origin (mx, my)
            # Color based on Distance of Correction?
            
            dx = mx - bx
            dy = my - by
            dist = np.hypot(dx, dy)
            
            alpha = min(0.3 + (np.log(n)/10), 0.9) # More opaque if more samples
            
            # Plot
            ax.arrow(bx, by, dx, dy, 
                     head_width=1.0, head_length=1.5, fc='red', ec='red', 
                     alpha=alpha, length_includes_head=True)
            
            # Dot at Block
            ax.scatter(bx, by, s=10, c='black', alpha=0.5)
            
            count += 1
        except Exception as e:
            continue
            
    ax.set_title(f"{title}\n(Arrows point from Block Location to Imputed Origin)")
    
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path} ({count} vectors)")
    plt.close()

def main():
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'puck', 'data')
    
    # Forward Model
    path_f = os.path.join(base_dir, 'blocked_shot_model_F.json')
    out_f = os.path.join(os.path.dirname(__file__), '..', 'analysis', 'blocked_shots', 'imputation_vectors_F.png')
    plot_model_vectors(path_f, "Forward Blocked Shot Imputation Vectors", out_f)
    
    # Defense Model
    path_d = os.path.join(base_dir, 'blocked_shot_model_D.json')
    out_d = os.path.join(os.path.dirname(__file__), '..', 'analysis', 'blocked_shots', 'imputation_vectors_D.png')
    plot_model_vectors(path_d, "Defenseman Blocked Shot Imputation Vectors", out_d)

if __name__ == "__main__":
    main()
