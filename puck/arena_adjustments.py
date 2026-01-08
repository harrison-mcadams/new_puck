
import json
import os
import math
from pathlib import Path

# Path to the data directory relative to this file
# puck/arena_adjustments.py -> ../data/arena_adjustments.json
BASE_DIR = Path(__file__).resolve().parent.parent
ADJUSTMENTS_FILE = BASE_DIR / "data" / "arena_adjustments.json"

_ADJUSTMENTS = None

def load_adjustments():
    global _ADJUSTMENTS
    if _ADJUSTMENTS is None:
        if ADJUSTMENTS_FILE.exists():
            try:
                with open(ADJUSTMENTS_FILE, 'r', encoding='utf-8') as f:
                    _ADJUSTMENTS = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load arena adjustments: {e}")
                _ADJUSTMENTS = {}
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

_ABB_TO_NAME = {
    "ANA": "Ducks", "ARI": "Coyotes", "BOS": "Bruins", "BUF": "Sabres",
    "CGY": "Flames", "CAR": "Hurricanes", "CHI": "Blackhawks", "COL": "Avalanche",
    "CBJ": "Blue Jackets", "DAL": "Stars", "DET": "Red Wings", "EDM": "Oilers",
    "FLA": "Panthers", "LAK": "Kings", "MIN": "Wild", "MTL": "Canadiens",
    "NSH": "Predators", "NJD": "Devils", "NYI": "Islanders", "NYR": "Rangers",
    "OTT": "Senators", "PHI": "Flyers", "PIT": "Penguins", "SJS": "Sharks",
    "SEA": "Kraken", "STL": "Blues", "TBL": "Lightning", "TOR": "Maple Leafs",
    "UTA": "Utah Hockey Club", "VAN": "Canucks", "VGK": "Golden Knights",
    "WSH": "Capitals", "WPG": "Jets"
}

def resolve_arena(name_or_abb):
    """Resolves a generic name (Full Name, Abb, or Short Name) to the standardized Short Name used in keys."""
    if name_or_abb in _TEAM_NAME_MAPPING:
        return _TEAM_NAME_MAPPING[name_or_abb]
    if name_or_abb in _ABB_TO_NAME:
        return _ABB_TO_NAME[name_or_abb]
    # Check if it's already a value (Short Name)
    if name_or_abb in _TEAM_NAME_MAPPING.values():
        return name_or_abb
    return name_or_abb

def adjust_shot(x, y, arena, season):
    """
    Adjusts shot coordinates based on Shuckers & Curro arena bias model.
    Only applies to shots; returns original coordinates if arena/season not found.
    
    Args:
        x (float): X coordinate (ft)
        y (float): Y coordinate (ft)
        arena (str): Home team name (e.g., 'Rangers', 'NYR')
        season (str or int): Season ID (e.g. 20232024)
        
    Returns:
        (adj_x, adj_y)
    """
    arena = resolve_arena(arena)
    
    if x is None: return x, y
    
    adj_map = load_adjustments()
    
    season = str(season)
    # If season strictly not found, try to fallback to nearest previous season?
    # For now, simplistic fallback to most recent key if exact match missing
    # (or maybe strict persistence is better? Using most recent is dangerous if bias shifts)
    # Let's stick to strict or simple max fallback.
    if season not in adj_map:
        if not adj_map:
            return x, y
        # Fallback to latest available (assuming bias persists)
        season = max(adj_map.keys())

    season_data = adj_map.get(season, {})
    
    # Resolve Arena
    # The JSON uses Team Name as key (e.g. "Rangers")
    if arena not in season_data:
        return x, y
        
    arena_data = season_data[arena]
    
    # Apply X Adjustment
    # Key is integer string of abs(x)
    adj_x = x
    try:
        abs_x = abs(x)
        lookup_x = str(int(round(abs_x)))
        if lookup_x in arena_data.get('x', {}):
            delta_x = arena_data['x'][lookup_x]
            # Add delta to absolute value
            new_abs_x = abs_x + delta_x
            adj_x = math.copysign(new_abs_x, x)
    except Exception:
        pass
        
    # Apply Y Adjustment
    adj_y = y
    try:
        if y is not None:
            abs_y = abs(y)
            lookup_y = str(int(round(abs_y)))
            if lookup_y in arena_data.get('y', {}):
                delta_y = arena_data['y'][lookup_y]
                # Add delta to absolute value
                new_abs_y = abs_y + delta_y
                adj_y = math.copysign(new_abs_y, y)
    except Exception:
        pass
        
    return adj_x, adj_y
