
import sys
import os

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print(f"Path: {sys.path[-1]}")

try:
    from puck import parse
    print("Imported puck.parse successfully.")
    
    # Test the relative import path logic
    try:
        from puck import rink
        print("Imported puck.rink successfully.")
    except Exception as e:
        print(f"Failed to import puck.rink: {e}")

    print("Testing infer_home_defending_side_from_play...")
    try:
        # Pass minimal args to trigger import
        res = parse.infer_home_defending_side_from_play({})
        print(f"Result: {res}")
    except Exception as e:
        print(f"CRASHED: {e}")

except Exception as e:
    print(f"Failed to import puck.parse: {e}")
