
import json
import os
import math

# Load adjustments once at module level if possible, or lazy load
ADJUSTMENTS_FILE = os.path.join("data", "arena_adjustments.json")
_ADJUSTMENTS = None

def load_adjustments():
    global _ADJUSTMENTS
    if _ADJUSTMENTS is None:
        if os.path.exists(ADJUSTMENTS_FILE):
            with open(ADJUSTMENTS_FILE, 'r') as f:
                _ADJUSTMENTS = json.load(f)
        else:
            _ADJUSTMENTS = {}
    return _ADJUSTMENTS

_TEAM_NAME_MAPPING = {
    "Anaheim Ducks": "Ducks",
    "Arizona Coyotes": "Coyotes", 
    "Boston Bruins": "Bruins",
    "Buffalo Sabres": "Sabres",
    "Calgary Flames": "Flames",
    "Carolina Hurricanes": "Hurricanes",
    "Chicago Blackhawks": "Blackhawks",
    "Colorado Avalanche": "Avalanche",
    "Columbus Blue Jackets": "Blue Jackets",
    "Dallas Stars": "Stars",
    "Detroit Red Wings": "Red Wings",
    "Edmonton Oilers": "Oilers",
    "Florida Panthers": "Panthers",
    "Los Angeles Kings": "Kings",
    "Minnesota Wild": "Wild",
    "Montreal Canadiens": "Canadiens",
    "Nashville Predators": "Predators",
    "New Jersey Devils": "Devils",
    "New York Islanders": "Islanders",
    "New York Rangers": "Rangers",
    "Ottawa Senators": "Senators",
    "Philadelphia Flyers": "Flyers",
    "Pittsburgh Penguins": "Penguins",
    "San Jose Sharks": "Sharks",
    "Seattle Kraken": "Kraken",
    "St. Louis Blues": "Blues",
    "Tampa Bay Lightning": "Lightning",
    "Toronto Maple Leafs": "Maple Leafs",
    "Utah Hockey Club": "Utah Hockey Club",
    "Vancouver Canucks": "Canucks",
    "Vegas Golden Knights": "Golden Knights",
    "Washington Capitals": "Capitals",
    "Winnipeg Jets": "Jets"
}

def adjust_shot(x, y, arena, season):
    """
    Adjusts shot coordinates based on Shuckers & Curro arena bias model.
    xy: Original coordinates usually from -100 to 100 and -42 to 42.
    arena: Name of the home team or venue (must match training key).
    season: Season ID (e.g., '20232024').
    
    Returns: (adj_x, adj_y)
    """
    # Normalize Arena Name
    if arena in _TEAM_NAME_MAPPING:
        arena = _TEAM_NAME_MAPPING[arena]
    adj_map = load_adjustments()
    
    # 1. Resolve Season
    season = str(season)
    if season not in adj_map:
        # Fallback to latest available season if current not found?
        available_seasons = sorted(adj_map.keys())
        if available_seasons:
            season = available_seasons[-1] # Use most recent
        else:
            return x, y # No data at all

    season_data = adj_map[season]
    
    # 2. Resolve Arena
    if arena not in season_data:
        return x, y # No data for this arena
        
    arena_data = season_data[arena]
    
    # 3. Apply Adjustments
    # Logic: NewAbs = OldAbs + Delta(OldAbs)
    # NewVal = sign(OldVal) * NewAbs
    
    # Handle X
    if x is not None:
        abs_x = abs(x)
        lookup_x = str(int(round(abs_x))) # Lookup key is integer string
        
        # We might need to handle out of bounds or float inputs
        # Our training generated keys for 0..100.
        if lookup_x in arena_data.get('x', {}):
            delta_x = arena_data['x'][lookup_x]
            new_abs_x = abs_x + delta_x
            adj_x = math.copysign(new_abs_x, x)
        else:
            # If out of range (e.g. 101?), no adjustment or closest?
            # Usually shots aren't > 100.
            adj_x = x
    else:
        adj_x = x
        
    # Handle Y
    if y is not None:
        abs_y = abs(y)
        lookup_y = str(int(round(abs_y)))
        
        if lookup_y in arena_data.get('y', {}):
            delta_y = arena_data['y'][lookup_y]
            new_abs_y = abs_y + delta_y
            adj_y = math.copysign(new_abs_y, y)
        else:
            adj_y = y
    else:
        adj_y = y
        
    return adj_x, adj_y

if __name__ == "__main__":
    # Simple test
    print("Testing Adjustment...")
    # Mocking data file for testing locally if it doesn't exist yet
    test_adj = {
        "20232024": {
            "Test Arena": {
                "x": {"80": 2.5}, # At 80ft, add 2.5ft (adjust further from center)
                "y": {"0": 0.0}
            }
        }
    }
    
    if not os.path.exists(ADJUSTMENTS_FILE):
        with open(ADJUSTMENTS_FILE, 'w') as f:
            json.dump(test_adj, f)
            
    # Test
    # Input 80 (should become 82.5)
    ax, ay = adjust_shot(80, 0, "Test Arena", "20232024")
    print(f"80 -> {ax}")
    assert ax == 82.5
    
    # Input -80 (should become -82.5)
    ax, ay = adjust_shot(-80, 0, "Test Arena", "20232024")
    print(f"-80 -> {ax}")
    assert ax == -82.5
    
    print("Test Passed.")
