# Shift Mapping and Validation Tools

This document describes the tools and improvements for debugging and fixing player/team mapping mismatches between NHL API shifts and HTML-parsed shifts.

## Problem Statement

The NHL API and HTML shift reports sometimes produce different results due to:
1. **Missing data**: API may not return shifts for some games
2. **Mapping errors**: Jersey numbers need to be mapped to canonical NHL player IDs
3. **Team attribution**: Players may be attributed to wrong team (home vs away)
4. **Zero-length shifts**: Some API responses include shifts where start == end
5. **Name variations**: Player names may be formatted differently across sources

## Improvements Implemented

### 1. Zero-Length Shift Filtering

**File**: `nhl_api.py` - `get_shifts()` function

Filters out shifts where `start_seconds == end_seconds` to avoid noise in game state calculations.

```python
# Automatically removes zero-length shifts
all_shifts = filtered  # After filtering
```

### 2. Three-Tier Player ID Mapping

**File**: `nhl_api.py` - `get_shifts_from_nhl_html()` function

The mapping logic tries three strategies in order:

#### Strategy 1: Jersey Number → Player ID (Same Team)
Maps jersey number to canonical player ID using the detected team (home/away).

```python
canonical_id = roster_map.get(team_side, {}).get(player_number)
```

#### Strategy 2: Name-Based Mapping
If jersey mapping fails, tries to map using normalized player name.

```python
normalized = _normalize_name(player_name)
canonical_id = name_map.get(normalized)
```

Name normalization handles:
- Case insensitivity
- Punctuation removal (commas, apostrophes, quotes)
- Whitespace normalization
- **Unicode/accent handling** (ü → u, é → e, etc.)

#### Strategy 3: Cross-Team Jersey Mapping
If both fail, checks the *other* team's roster (in case team_side was wrong).
When a match is found, automatically corrects `team_side` and `team_id`.

```python
other_side = 'away' if team_side == 'home' else 'home'
canonical_id = roster_map.get(other_side, {}).get(player_number)
if canonical_id:
    shift['team_side'] = other_side
    shift['team_id'] = team_ids.get(other_side)
```

### 3. Team ID Inference

**File**: `nhl_api.py` - `get_shifts_from_nhl_html()` function

If a canonical player ID is found but team_id is missing, infers team_id by checking which roster contains the player.

```python
if canonical_id in roster_map.get('home', {}).values():
    shift['team_id'] = team_ids.get('home')
elif canonical_id in roster_map.get('away', {}).values():
    shift['team_id'] = team_ids.get('away')
```

### 4. HTML Fallback

**File**: `nhl_api.py` - `get_shifts()` function

When API endpoints fail to return shift data, automatically falls back to HTML parsing:

```python
if data is None:
    backup = get_shifts_from_nhl_html(game_id, force_refresh=force_refresh, debug=True)
    if backup and backup.get('all_shifts'):
        return backup
```

## Tools

### 1. Debug Parse Shifts (`scripts/debug_parse_shifts.py`)

Compares API and HTML shift sources and validates mappings.

**Usage**:
```bash
python scripts/debug_parse_shifts.py <game_id> [--no-save]
```

**Output**:
- Summary of API and HTML shift counts
- Validation results (missing team_id, player_id still as jersey number)
- Sample mismatches
- Debug info from HTML parsing
- Compare_shifts diff summary

**Example**:
```bash
python scripts/debug_parse_shifts.py 2025020339
```

### 2. Save Game Data (`scripts/save_game_data.py`)

Fetches and saves all intermediary data for offline analysis.

**Usage**:
```bash
python scripts/save_game_data.py <game_id> [--output-dir DIR]
```

**Saves**:
- `game_feed.json` - Full game feed from API
- `api_shifts_full.json` - Complete API shifts response
- `api_shifts_only.json` - Just the all_shifts array
- `html_shifts_full.json` - Complete HTML shifts response
- `html_shifts_only.json` - Just the all_shifts array
- `html_debug.json` - Debug info from HTML parsing
- `raw_html.txt` - Raw HTML content
- `roster_mapping.json` - Jersey number → player ID mappings
- `team_ids.json` - Team IDs for home/away
- `name_to_id_mapping.json` - Name → player ID mappings
- `summary.json` - Overview of all saved data

**Example**:
```bash
python scripts/save_game_data.py 2025020339 --output-dir game_data
```

### 3. Trace Interval (`scripts/trace_interval.py`)

Traces which shifts are active during a specific time interval to debug game state discrepancies.

**Usage**:
```bash
python scripts/trace_interval.py <game_id> <start_seconds> <end_seconds> [--source {api|html|both}]
```

**Output**:
- Lists all shifts overlapping the interval
- Shows player details (ID, name, number, team)
- Estimates game state (e.g., "5v6")
- Compares API vs HTML sources
- Identifies players present in only one source

**Example** (debug the 457s-540s interval where Timo Meier was incorrectly shown):
```bash
python scripts/trace_interval.py 2025020339 457 540
```

### 4. Compare Game State (`scripts/compare_game_state.py`)

Compares game_state and is_net_empty derived from API vs HTML shifts.

**Usage**:
```bash
python scripts/compare_game_state.py --game <game_id> [--debug]
```

**Output**:
- Shift counts for each source
- Game state comparison (overlap, mismatch seconds)
- Net empty comparison
- Sample mismatch intervals

**Example**:
```bash
python scripts/compare_game_state.py --game 2025020339
```

## Testing

### Run Mapping Logic Tests

**File**: `tests/test_mapping_logic.py`

Tests the core mapping logic without requiring network access.

```bash
python tests/test_mapping_logic.py
```

**Tests**:
- Name normalization (including Unicode)
- Zero-length shift filtering
- Three-tier mapping fallback logic
- Team ID inference
- Team side correction

All tests currently pass ✓

## Debugging Workflow

### Scenario: API and HTML produce different game states

1. **Save intermediary data**:
   ```bash
   python scripts/save_game_data.py 2025020339
   ```

2. **Run debug validation**:
   ```bash
   python scripts/debug_parse_shifts.py 2025020339
   ```
   
   Check validation output for:
   - Missing team_id counts
   - Player_id still as jersey number counts
   - Unmapped players

3. **Identify problematic interval**:
   
   If debug shows different game states (e.g., API shows 5v6 but HTML shows 5v5), note the time range.

4. **Trace the interval**:
   ```bash
   python scripts/trace_interval.py 2025020339 457 540
   ```
   
   This will show:
   - Which players API thinks are on ice
   - Which players HTML thinks are on ice
   - Which players appear in only one source

5. **Inspect saved data**:
   
   Open saved JSON files to examine:
   - `game_data/game_2025020339/api_shifts_only.json` - Raw API shifts
   - `game_data/game_2025020339/html_shifts_only.json` - Parsed HTML shifts
   - `game_data/game_2025020339/roster_mapping.json` - Jersey mappings
   - `game_data/game_2025020339/html_debug.json` - HTML parsing debug info

6. **Fix mapping issues**:
   
   If players are unmapped or incorrectly mapped:
   - Check if roster_mapping has the player
   - Check if name_to_id_mapping has the player
   - Consider adding player to mapping source
   - Check for typos or name format differences

## Key Functions

### `_normalize_name(name: str) -> str`
Normalizes player name for consistent matching:
- Lowercase
- Removes punctuation (', ", comma)
- Normalizes Unicode (ü → u)
- Collapses whitespace

### `_build_name_to_id_map(game_id) -> Dict[str, int]`
Builds name → player_id mapping from game feed.

### `_get_roster_mapping(game_id) -> Dict[str, Dict[int, int]]`
Extracts jersey number → player_id mapping per team.

Returns: `{'home': {12: 8471675, ...}, 'away': {12: 8475798, ...}}`

### `_get_team_ids(game_id) -> Dict[str, int]`
Extracts team IDs for home and away teams.

Returns: `{'home': 1, 'away': 6}`

## Validation Criteria

For shifts to be considered correctly mapped:

1. **team_id must be set**: Every shift must have a team_id (not None)
2. **player_id must be canonical**: player_id should be an 8-digit NHL ID (e.g., 8471675), not a jersey number (1-99)

The `debug_parse_shifts.py` script validates these criteria and reports:
- `missing_team_id`: Count of shifts without team_id
- `player_id_is_jersey`: Count of shifts where player_id looks like a jersey number

**Success criteria**: Both counts should be 0.

## Known Limitations

1. **Network dependency**: Most tools require network access to fetch game data. Use `save_game_data.py` to download data for offline analysis.

2. **Unicode in test**: The test expects 'muller' but we preserve 'müller' in some contexts. This is acceptable as it doesn't affect matching (normalization handles it).

3. **Goalie classification**: The `trace_interval.py` tool counts all players as skaters. For accurate game state, we'd need to classify goalies vs skaters (this is done in `timing_new.py`).

## Environment

- Python 3.x
- Dependencies: requests, BeautifulSoup4, pandas, numpy (see requirements.txt)
- No API keys required (uses public NHL endpoints)

## Next Steps

If validation still shows issues:

1. **Check HTML parsing**: Verify tables_scanned, players_scanned in debug output
2. **Verify roster completeness**: Ensure all players are in roster_mapping
3. **Add name aliases**: If names differ significantly, add to name_map manually
4. **Check API accuracy**: Compare against official NHL game reports
5. **Report upstream**: If API data is incorrect, consider reporting to NHL

## Files Modified

- `nhl_api.py`: Core improvements to get_shifts and get_shifts_from_nhl_html
- `scripts/debug_parse_shifts.py`: Fixed imports, enhanced validation
- `scripts/save_game_data.py`: New tool to save intermediary data
- `scripts/trace_interval.py`: New tool to trace specific intervals
- `tests/test_mapping_logic.py`: New comprehensive tests

## Summary

These improvements ensure that:
- Zero-length shifts are filtered out
- Player jersey numbers are mapped to canonical NHL player IDs
- Team IDs are correctly assigned
- HTML fallback works when API fails
- Debugging tools are available for offline analysis

The three-tier mapping fallback ensures maximum coverage, and the tracing tools make it easy to identify and fix remaining discrepancies.
