"""features.py

Central repository for defining named sets of features for the xG models.
This allows us to easily swap, compare, and version feature configurations.
"""

from typing import List, Dict

# Basic coordinates
COORDINATES = ['distance', 'angle_deg']

# Game situation
# Note: game_state is listed here as a feature for base xG models. The
# mixed-effects model uses game_state as a *splitting variable* and handles
# it separately — it does not draw from these feature sets.
SITUATION = ['game_state', 'score_diff', 'period_number', 'time_elapsed_in_period_s', 'total_time_elapsed_s']
SHOT_TYPE = ['shot_type']
HANDEDNESS = ['shoots_catches']
PLAYER_ROLE = ['shooter_role']

# prior event
REBOUND = ['is_rebound', 'rebound_angle_change', 'rebound_time_diff']
RUSH = ['is_rush']
PRIOR_EVENT = ['last_event_type', 'last_event_time_diff', 'dist_from_last_event', 'speed_from_last_event', 'angle_change_last_event']

# Named Feature Sets
FEATURE_SETS = {
    'minimal': COORDINATES,
    'baseline': COORDINATES + ['game_state'] + SHOT_TYPE + HANDEDNESS + PLAYER_ROLE,
    'standard': COORDINATES + SITUATION + REBOUND + RUSH + PRIOR_EVENT + PLAYER_ROLE,
    'all_inclusive': COORDINATES + SITUATION + SHOT_TYPE + HANDEDNESS + REBOUND + RUSH + PRIOR_EVENT + PLAYER_ROLE,
}

def get_features(name: str = 'standard') -> List[str]:
    """Retrieve a feature set by name."""
    if name not in FEATURE_SETS:
        print(f"Warning: Feature set '{name}' not found. Defaulting to 'standard'.")
        return FEATURE_SETS['standard']
    return FEATURE_SETS[name]

def list_feature_sets() -> Dict[str, List[str]]:
    """Return all available feature sets."""
    return FEATURE_SETS
