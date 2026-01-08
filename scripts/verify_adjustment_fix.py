
import sys
from pathlib import Path

# Ensure we can import from local puck package
sys.path.append(str(Path.cwd()))

try:
    from puck.arena_adjustments import adjust_shot
except ImportError:
    print("Could not import puck.arena_adjustments. Make sure you are in the root of the repo.")
    sys.exit(1)
    
def test_adjustment():
    # Test case: New York Rangers (should map to Rangers)
    # Rangers have adjustments, e.g. input 80 -> should change
    
    # Check if we have adjustment data for Rangers in 20232024
    # (Assuming we saw it in the debug output)
    
    x_in = 80.0
    y_in = 0.0
    season = "20232024"
    arena_full = "New York Rangers"
    
    x_adj, y_adj = adjust_shot(x_in, y_in, arena_full, season)
    
    print(f"Input: ({x_in}, {y_in}) | Arena: '{arena_full}'")
    print(f"Output: ({x_adj}, {y_adj})")
    
    if x_adj == x_in:
        print("FAIL: No adjustment applied (x_adj == x_in)")
        sys.exit(1)
    else:
        print(f"SUCCESS: Adjustment applied. Delta: {x_adj - x_in}")

if __name__ == "__main__":
    test_adjustment()
