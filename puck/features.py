"""features.py

Central repository for defining named sets of features for the xG models.
This allows us to easily swap, compare, and version feature configurations.
"""

from typing import List, Dict

# Basic coordinates
COORDINATES = ['distance', 'angle_deg']

# Game situation
SITUATION = ['game_state', 'score_diff', 'period_number', 'time_elapsed_in_period_s', 'total_time_elapsed_s']
SHOT_TYPE = ['shot_type']
HANDEDNESS = ['shoots_catches']

# prior event
# - rebounds
# - cycle
# - rush

# Named Feature Sets
FEATURE_SETS = {
    'minimal': COORDINATES,
    'baseline': COORDINATES + ['game_state'],
    'standard': COORDINATES + SITUATION,
    'all_inclusive': COORDINATES + SITUATION + SHOT_TYPE + HANDEDNESS,
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
