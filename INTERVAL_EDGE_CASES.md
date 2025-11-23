# Interval Filtering Edge Case Handling

## Overview

The `_apply_intervals` function in `analyze.py` filters events based on time intervals computed by the timing module. This document describes how edge cases are handled, particularly for events that occur at exact boundary times between intervals.

## The Problem

Events can occur at exact boundary times where the game state changes. For example:
- A power-play goal occurs at time `t=100.0` seconds
- At `t=100.0`, the game state transitions from `5v4` (power play) to `5v5` (even strength)
- The timing module computes intervals: `[50, 100]` for `5v4` and `[100, 150]` for `5v5`

Without proper validation, a naive time-interval filter might:
1. Include the goal in both intervals (if using `>=` and `<=`)
2. Incorrectly attribute the goal to `5v5` when it actually occurred during `5v4`
3. Miss events that should be included based on their actual state

## The Solution

After selecting events by time intervals, `_apply_intervals` validates each matched event against the requested condition:

### Implementation

1. **Time-based filtering**: First, collect all events whose timestamps fall within the computed intervals
   ```python
   mask = times.notna() & (times >= start) & (times <= end)
   ```

2. **Post-filter validation**: When condition includes state-based filters (`game_state` or `is_net_empty`):
   - Compute `game_state_relative_to_team` for matched events using `timing.add_game_state_relative_column`
   - Build a mask using `parse.build_mask` to test the condition
   - Keep only events that satisfy both time-interval membership AND condition criteria

### Code Location

File: `analyze.py`  
Function: `_apply_intervals`  
Lines: ~600-650 (post-filter validation block)

## Behavior Details

### Game State Validation

When the condition includes `{'game_state': ['5v5']}`:

1. All events in time intervals are collected
2. For each event, compute its `game_state_relative_to_team` (e.g., '5v4', '5v5', '4v5')
3. Filter to keep only events where `game_state_relative_to_team` matches the requested states

**Example**: Event at boundary time `t=100` with `game_state='5v4'` is excluded from a `5v5` filter, even if the interval `[100, 150]` is labeled as a `5v5` interval.

### is_net_empty Validation

When the condition includes `{'is_net_empty': [0]}`:

1. All events in time intervals are collected
2. Each event's `is_net_empty` attribute is checked
3. Filter to keep only events where `is_net_empty` matches the requested values

**Example**: Event at `t=200` with `is_net_empty=1` is excluded from a filter requesting `is_net_empty=0`.

### Combined Conditions

When multiple conditions are specified (e.g., `{'game_state': ['5v5'], 'is_net_empty': [0]}`):
- Events must satisfy ALL conditions (AND logic)
- Both time-interval membership and per-event attribute checks must pass

## Convention and Edge Case Rules

1. **Boundary Time Handling**: Events at exact interval boundaries are validated by their per-event attributes, not by which interval they fall into
   
2. **game_state Computation**: Uses `timing.add_game_state_relative_column` which:
   - Computes game state relative to the selected team
   - Handles penalty events specially (inherits state from nearby faceoffs)
   - Flips state for opponent events (e.g., '5v4' becomes '4v5')

3. **Tie-Breaking**: When an event occurs exactly at a boundary:
   - The event's own `game_state` attribute determines inclusion
   - If `game_state` is missing/null for penalties, timing module infers it from nearby faceoffs

4. **Performance**: Post-validation only runs when needed:
   - Skipped if condition doesn't include `game_state` or `is_net_empty`
   - Applied per-game to minimize overhead
   - Uses vectorized operations where possible

## Testing

Tests are located in `tests/test_interval_edge_cases.py`:

1. **test_boundary_event_game_state_validation**: Validates that a goal at a 5v4→5v5 transition is correctly excluded from a 5v5 filter
2. **test_is_net_empty_validation**: Validates that events are filtered by is_net_empty at boundaries

Run tests:
```bash
python tests/test_interval_edge_cases.py
```

Expected output:
```
✓ PASS: Goal at time 100 (5v4) correctly excluded from 5v5 filter
✓ PASS: Faceoff at time 100 (5v5) correctly included in 5v5 filter
✓ PASS: Event at time 200 (is_net_empty=1) correctly excluded
✓ All tests passed!
```

## Future Enhancements

Potential improvements for edge case handling:

1. **Configurable boundary behavior**: Allow users to specify whether boundary events should be included/excluded
2. **Timestamp precision**: Handle sub-second timing resolution more carefully
3. **Additional state attributes**: Extend validation to other per-event attributes (e.g., score_state, zone)
4. **Performance optimization**: Cache computed game_state columns across multiple condition tests

## Related Documentation

- `timing.py`: `add_game_state_relative_column` function
- `parse.py`: `build_mask` function  
- `analyze.py`: `xgs_map` and `_apply_intervals` functions
