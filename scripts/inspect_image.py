import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def rgb2hex(r, g, b):
    return "#{:02x}{:02x}{:02x}".format(r, g, b)

img_path = "analysis/league/20252026/5v5/PHI_relative.png"
try:
    img = Image.open(img_path)
    data = np.array(img)
    print(f"Image Shape: {data.shape}")
    
    # Check Top-Left Corner (Corner of Axes Box)
    # usually rink is centered. corners are clearly outside the rounded rink.
    # Top Left: [0, 0]
    # Check a 10x10 patch
    patch = data[0:10, 0:10]
    
    # Calculate average color
    avg_color = np.mean(patch, axis=(0,1))
    print(f"Top-Left Corner Avg Color: {avg_color}")
    if len(avg_color) >= 3:
        print(f"Hex: {rgb2hex(int(avg_color[0]), int(avg_color[1]), int(avg_color[2]))}")
        
    # Check if there's variation
    unique_colors = np.unique(patch.reshape(-1, data.shape[2]), axis=0)
    print(f"Unique colors in top-left patch: {len(unique_colors)}")
    for c in unique_colors[:5]:
        print(f"  {c}")
        
    # Check Rink Surface (Left Zone)
    h, w, _ = data.shape
    center_patch = data[h//2-5:h//2+5, w//4-5:w//4+5] # Left side of rink
    avg_center = np.mean(center_patch, axis=(0,1))
    print(f"Rink Surface (Left Zone) Avg: {avg_center}")
    if len(avg_center) >= 3:
        print(f"Hex: {rgb2hex(int(avg_center[0]), int(avg_center[1]), int(avg_center[2]))}")
        
    # Check "Corner" Region (Inside Box, Outside Rounded Rink)
    # Right end of rink is at image width ~90%?
    # Top edge is at height ~10%?
    # Let's sample discrete points to find the "off-white"
    
    print("Checking specific points for Off-White:")
    # Scan diagonal from Top-Right corner inwards
    found_off_white = False
    for i in range(50): # 50 steps
        x = w - 10 - i*5
        y = 10 + i*5
        if x < 0 or y >= h: break
        
        px = data[y, x]
        # Ignore if transparent or pure white
        if np.all(px == 255): continue
        if px[3] == 0: continue # Transparent
        
        print(f"Non-White Pixel at ({x}, {y}): {px}")
        print(f"Hex: {rgb2hex(px[0], px[1], px[2])}")
        found_off_white = True
        break
        
    if not found_off_white:
        print("No non-white pixels found in top-right scan.")

except Exception as e:
    print(f"Error: {e}")
