"""Compatibility shim for timing_brainstorming.

This module re-exports the public API from the refactored `timing.py` and
emits a DeprecationWarning so callers migrate to importing `timing`.
"""
from warnings import warn
warn("'timing_brainstorming' has been renamed to 'timing' — importing shim.", DeprecationWarning)

from . import timing as _timing  # type: ignore

# Re-export selected symbols for backward compatibility
load_season_df = _timing.load_season_df
select_team_game = _timing.select_team_game
intervals_for_condition = _timing.intervals_for_condition
intervals_for_conditions = _timing.intervals_for_conditions
add_game_state_relative_column = _timing.add_game_state_relative_column
demo_for_export = _timing.demo_for_export

__all__ = [
    'load_season_df', 'select_team_game', 'intervals_for_condition',
    'intervals_for_conditions', 'add_game_state_relative_column', 'demo_for_export'
]
