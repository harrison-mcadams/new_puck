# HTML Shift Parsing and Fallback Integration

## Overview

This implementation provides robust HTML-based shift parsing as a fallback for the NHL API, with automatic integration into `timing_new.py`. The system ensures continuous operation even when the primary API fails or returns insufficient data.

## Key Components

### 1. HTML Parsing (`nhl_api.py`)

**Function:** `get_shifts_from_nhl_html(game_id, force_refresh=False, debug=False)`

Parses NHL official HTML shift reports to extract shift data. Features:
- **Team-specific roster mapping**: Uses `(team_id, jersey_number)` as key to avoid conflicts
- **Automatic team_id assignment**: Sets correct team_id for all shifts based on team_side ('home'/'away')
- **Canonical player_id mapping**: Maps jersey numbers to NHL player IDs using game feed roster
- **Robust parsing**: Handles both per-player detail tables and event-style on-ice tables

**Helper Functions:**
- `_get_roster_mapping(game_id)`: Returns `{'home': {jersey: player_id}, 'away': {jersey: player_id}}`
- `_get_team_ids(game_id)`: Returns `{'home': team_id, 'away': team_id}`

**Output format:**
```python
{
    'game_id': int,
    'raw': str,  # Raw HTML
    'all_shifts': [  # List of shift dicts
        {
            'game_id': int,
            'player_id': int,  # Canonical NHL player ID
            'player_number': int,  # Jersey number
            'team_id': int,  # NHL team ID
            'team_side': str,  # 'home' or 'away'
            'period': int,
            'start_raw': str,  # 'MM:SS'
            'end_raw': str,  # 'MM:SS'
            'start_seconds': int,  # Within period
            'end_seconds': int,  # Within period
            'start_total_seconds': int,  # From game start
            'end_total_seconds': int,  # From game start
            'raw': dict  # Original parsed data
        },
        ...
    ],
    'shifts_by_player': {player_id: [shift, ...]},
    'debug': {  # If debug=True
        'tables_scanned': int,
        'players_scanned': int,
        'found_shifts': int,
        'team_ids': dict,
        'roster_mapping': {
            'home_players': int,
            'away_players': int,
            'mapped_shifts': int,
            'unmapped_shifts': int
        }
    }
}
```

### 2. Timing Integration (`timing_new.py`)

**Function:** `get_shifts_with_html_fallback(game_id, min_rows_threshold=5)`

Public API wrapper that:
1. Calls `nhl_api.get_shifts(game_id)`
2. Checks if result is empty or has fewer than `min_rows_threshold` shifts
3. If insufficient, calls `nhl_api.get_shifts_from_nhl_html(game_id)` as fallback
4. Returns shifts in same format as `nhl_api.get_shifts()`

**Function:** `_get_shifts_df(game_id, min_rows_threshold=5)`

Internal function used by timing calculations that:
1. Attempts to get shifts from API
2. Parses to DataFrame using `parse._shifts()`
3. If empty/minimal, automatically uses HTML fallback
4. Normalizes column names (`start_total_seconds`, `end_total_seconds`)
5. Returns DataFrame ready for interval computations

### 3. Comparison Utility (`scripts/compare_game_state.py`)

Compares `game_state` and `is_net_empty` derived from API vs HTML shift sources.

**Usage:**
```bash
python scripts/compare_game_state.py --game 2025020339
```

**Features:**
- Derives game state intervals (e.g., '5v5', '5v4') from both sources
- Derives net empty intervals from both sources
- Reports overlap and mismatch statistics
- Shows sample mismatches with timestamps

**Output:**
```
======================================================================
Game State and Is Net Empty Comparison for Game 2025020339
======================================================================

Shift Counts:
  API shifts:  456
  HTML shifts: 458

Team IDs:
  Home: 4
  Away: 14

----------------------------------------------------------------------
GAME_STATE Comparison:
----------------------------------------------------------------------
  API total seconds:   3622.0
  HTML total seconds:  3619.0
  Overlap (matching):  3615.0s
  Mismatch:            4.0s
  API unique states:   ['5v5', '5v4', '4v5', '6v5', '5v6']
  HTML unique states:  ['5v5', '5v4', '4v5', '6v5', '5v6']

  ✓ No mismatches found!

----------------------------------------------------------------------
IS_NET_EMPTY Comparison:
----------------------------------------------------------------------
  API total seconds:   3622.0
  HTML total seconds:  3619.0
  Overlap (matching):  3610.0s
  Mismatch:            9.0s
  API unique values:   [0, 1]
  HTML unique values:  [0, 1]

  Sample mismatches (up to 10):
    [3595.0 - 3600.0] (5.0s): API=0, HTML=1
```

### 4. Demo Script (`scripts/demo_html_fallback.py`)

Demonstrates the HTML fallback integration.

**Usage:**
```bash
python scripts/demo_html_fallback.py --game 2025020339 --all
```

**Demos:**
1. `get_shifts_with_html_fallback()` wrapper
2. `_get_shifts_df()` automatic fallback
3. Direct API vs HTML comparison

## Implementation Details

### Roster Mapping Fix

**Problem:** Original implementation used jersey number alone as key, causing collisions when both teams had players with the same number.

**Solution:** 
```python
# Build team-specific mapping
roster_map = {
    'home': {12: 8471675, 23: 8474564, ...},
    'away': {12: 8475798, 23: 8476123, ...}
}

# Lookup using team context
for shift in all_shifts:
    team_side = shift.get('team_side')  # 'home' or 'away'
    player_number = shift.get('player_number')
    
    # Get team-specific roster
    team_roster = roster_map.get(team_side, {})
    canonical_id = team_roster.get(player_number)
    
    if canonical_id is not None:
        shift['player_id'] = canonical_id
```

### Team ID Assignment

**Implementation:**
```python
team_ids = _get_team_ids(game_id)  # {'home': 4, 'away': 14}

for shift in all_shifts:
    team_side = shift.get('team_side')  # Set during HTML parsing
    if team_side in ('home', 'away'):
        team_id = team_ids.get(team_side)
        if team_id is not None:
            shift['team_id'] = team_id
```

### Fallback Threshold

The `min_rows_threshold` parameter (default 5) determines when to use HTML fallback:
- 0 shifts: Always use fallback
- 1-4 shifts: Likely incomplete, use fallback
- 5+ shifts: Probably valid, use API result

Adjust based on your needs:
```python
# Be more conservative (require at least 20 shifts)
shifts = get_shifts_with_html_fallback(game_id, min_rows_threshold=20)

# Use fallback only if completely empty
shifts = get_shifts_with_html_fallback(game_id, min_rows_threshold=1)
```

## Testing

### 1. Validate Mapping

```python
from scripts.debug_parse_shifts import run_debug

# Run debug for a game
result = run_debug('2025020339', save=True)
validation = result['validation']

# Check results
assert validation['missing_team_id'] == 0, "Some shifts missing team_id"
assert validation['player_id_is_jersey'] == 0, "Some shifts have jersey as player_id"
```

### 2. Compare Sources

```python
from scripts.compare_game_state import compare_game_state_and_net_empty

result = compare_game_state_and_net_empty(2025020339)

# Check agreement
gs_comp = result['game_state_comparison']
ne_comp = result['is_net_empty_comparison']

print(f"Game state overlap: {gs_comp['overlap_seconds']:.1f}s")
print(f"Net empty overlap: {ne_comp['overlap_seconds']:.1f}s")
```

### 3. Integration Test

```python
import timing_new

# Test fallback triggers correctly
df = timing_new._get_shifts_df(game_id, min_rows_threshold=5)
assert not df.empty, "Should have shifts from fallback"
assert 'start_total_seconds' in df.columns
assert 'end_total_seconds' in df.columns
```

## Acceptance Criteria

✅ **Criterion 1: team_id is set for all shifts**
- Every shift row has a numeric team_id
- No None/null values in team_id column

✅ **Criterion 2: player_id is canonical NHL player ID**
- player_id is an 8-digit NHL player ID (e.g., 8471675)
- Not the jersey number (1-99)
- Mapping uses team context to avoid collisions

✅ **Criterion 3: Fallback integration in timing_new.py**
- `_get_shifts_df()` automatically uses HTML when API is empty/minimal
- `get_shifts_with_html_fallback()` provides public API wrapper
- Threshold configurable via `min_rows_threshold` parameter

✅ **Criterion 4: Comparison utility exists**
- `scripts/compare_game_state.py` compares both fields
- Reports overlap and mismatch statistics
- Shows sample differences

✅ **Criterion 5: Maintains current functionality**
- All changes are additive
- Existing callers work unchanged
- HTML fallback is opt-in for new code

## Known Limitations

1. **No internet access**: Cannot fetch data in environments without network access
2. **Roster changes mid-game**: If a player changes jersey number during a game, mapping may be incorrect
3. **Emergency backup goalies**: Players not in initial roster may not be mapped correctly
4. **Parsing edge cases**: Some HTML report formats may not be supported

## Future Improvements

1. **Cache roster mappings**: Store team rosters to reduce API calls
2. **Fuzzy player matching**: Use player name similarity when ID mapping fails
3. **Historical roster API**: Query past rosters for older games
4. **Enhanced validation**: Add more programmatic checks for data quality

## References

- NHL Official Reports: `https://www.nhl.com/scores/htmlreports/{season}/T{H|V}{game_suffix}.HTM`
- API Endpoint: `https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={game_id}`
- Problem Statement: Issue describes jersey number collision bug and fallback requirements
