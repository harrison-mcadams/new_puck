import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from puck.arena_adjustments import adjust_shot

if __name__ == "__main__":
    # Simple test wrapper using the library function
    print("Testing Adjustment (via Library)...")
    
    # Test valid case
    ax, ay = adjust_shot(80, 0, "Rangers", "20232024")
    print(f"Rangers 2023 80,0 -> {ax},{ay}")
    
    # Test abbreviation
    ax2, ay2 = adjust_shot(80, 0, "NYR", "20232024")
    print(f"NYR 2023 80,0 -> {ax2},{ay2}")
    
    assert ax == ax2
    print("Test Passed.")
