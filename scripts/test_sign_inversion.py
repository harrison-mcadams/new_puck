
import json
import os
import math

def test_sign():
    # Load raw data
    path = os.path.join("data", "arena_adjustments.json")
    with open(path, 'r') as f:
        data = json.load(f)

    # Tampa 2018-2019
    tampa_data = data.get("20182019", {}).get("Lightning", {})
    x_map = tampa_data.get('x', {})
    
    # Check X=85
    # The key in JSON is likely "85" or "85.0"
    val_at_85 = x_map.get("85") or x_map.get("85.0")
    
    print(f"Retrieving Tampa 2018-19 Adjustment for X=85...")
    print(f"Raw Value in JSON: {val_at_85}")
    
    if val_at_85 is not None:
        delta = float(val_at_85)
        x_obs = 85.0
        
        # Current Logic (Add)
        x_add = x_obs + delta
        
        # Proposed Logic (Subtract)
        x_sub = x_obs - delta
        
        print(f"\nScenario: Observed Shot at X={x_obs} (Crease area)")
        print(f"Method A (Current - Add): {x_obs} + {delta} = {x_add}")
        print(f"  -> Result Location: {x_add} (Behind Net/Boards)")
        print(f"Method B (Proposed - Sub): {x_obs} - {delta} = {x_sub}")
        print(f"  -> Result Location: {x_sub} (High Slot)")
        
        print("\nPhysical Interpretation:")
        if x_add > 89:
            print("  Method A pushes the shot BEHIND the goal line (Invalidates shot).")
        if x_sub < 89:
            print("  Method B keeps the shot in the offensive zone, just further out.")

if __name__ == "__main__":
    test_sign()
