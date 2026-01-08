
import sys
import os
sys.path.append(os.getcwd())
from puck import arena_adjustments

# Force load
adj = arena_adjustments.load_adjustments()
print(f"Loaded adjustments for {len(adj)} seasons.")
if '20232024' in adj:
    print(f"20232024 has {len(adj['20232024'])} arenas.")
    # Pick an arena
    arena = list(adj['20232024'].keys())[0]
    print(f"Testing Arena: {arena}")
    x, y = 50.0, 0.0
    xa, ya = arena_adjustments.adjust_shot(x, y, arena, '20232024')
    print(f"Adjusted ({x}, {y}) -> ({xa}, {ya})")
else:
    print("20232024 not found in adjustments.")
