
import json
import os

def check_zones():
    path = os.path.join("data", "arena_adjustments.json")
    with open(path, 'r') as f:
        data = json.load(f)

    print("Checking for large adjustments (>8ft) in the Defensive/Offensive Zones (X > 60)...")
    
    warnings = []

    for season, s_data in data.items():
        if season == "20252026": continue 
        for arena, a_data in s_data.items():
            x_biases = a_data.get('x', {})
            
            for x_str, bias in x_biases.items():
                x_val = abs(float(x_str))
                b_val = abs(float(bias))
                
                if b_val > 8.0:
                    description = "Neutral Zone"
                    if x_val > 75: description = "Deep Zone / Net Front"
                    elif x_val > 60: description = "High Slot / Blue Line"
                    elif x_val > 25: description = "Neutral / Blue Line"
                    else: description = "Center Ice"

                    if x_val > 50: # Only care about offensive half for xG mostly
                         warnings.append(f"{season} {arena}: Bias {b_val} at X={x_val} ({description})")

    if warnings:
        for w in warnings[:20]:
             print(w)
        if len(warnings) > 20: print(f"... and {len(warnings)-20} more.")
    else:
        print("No large adjustments (>8ft) found in X > 50 zones.")

if __name__ == "__main__":
    check_zones()
