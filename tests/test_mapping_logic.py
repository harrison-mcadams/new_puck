#!/usr/bin/env python3
"""Test the mapping logic with sample data (no network required).

This test creates mock game feed and HTML data to verify that the mapping logic works correctly.
"""

import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import nhl_api


def test_normalize_name():
    """Test name normalization."""
    print("Testing _normalize_name()...")
    
    test_cases = [
        ("John Smith", "john smith"),
        ("O'Reilly, Ryan", "oreilly ryan"),
        ("Müller, Thomas", "muller thomas"),
        ("  Multiple   Spaces  ", "multiple spaces"),
        ("Name-With-Hyphens", "name-with-hyphens"),
        ("D'Angelo", "dangelo"),
    ]
    
    for input_name, expected in test_cases:
        result = nhl_api._normalize_name(input_name)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{input_name}' -> '{result}' (expected: '{expected}')")
        if result != expected:
            print(f"     FAILED: got '{result}' expected '{expected}'")
    
    print()


def test_zero_length_filtering():
    """Test that zero-length shifts are filtered out."""
    print("Testing zero-length shift filtering...")
    
    # Create mock shifts data with some zero-length shifts
    mock_data = {
        'data': [
            {'playerId': 8471675, 'teamId': 1, 'period': 1, 'start': '0:00', 'end': '1:30'},
            {'playerId': 8471676, 'teamId': 1, 'period': 1, 'start': '1:30', 'end': '1:30'},  # zero-length
            {'playerId': 8471677, 'teamId': 1, 'period': 1, 'start': '2:00', 'end': '3:00'},
            {'playerId': 8471678, 'teamId': 1, 'period': 1, 'start': '5:00', 'end': '5:00'},  # zero-length
        ]
    }
    
    # Manually parse similar to get_shifts logic
    all_shifts = []
    for entry in mock_data['data']:
        # Simple mm:ss parser
        def parse_mmss(s):
            parts = s.split(':')
            return int(parts[0]) * 60 + int(parts[1])
        
        start_sec = parse_mmss(entry['start'])
        end_sec = parse_mmss(entry['end'])
        
        shift = {
            'player_id': entry['playerId'],
            'start_seconds': start_sec,
            'end_seconds': end_sec,
        }
        all_shifts.append(shift)
    
    print(f"  Before filtering: {len(all_shifts)} shifts")
    
    # Apply zero-length filtering logic
    filtered = []
    for s in all_shifts:
        ss = s.get('start_seconds')
        es = s.get('end_seconds')
        if ss is not None and es is not None and float(ss) == float(es):
            continue
        filtered.append(s)
    
    print(f"  After filtering: {len(filtered)} shifts")
    print(f"  Removed: {len(all_shifts) - len(filtered)} zero-length shifts")
    
    # Verify
    if len(filtered) == 2 and len(all_shifts) - len(filtered) == 2:
        print("  ✓ Zero-length filtering works correctly")
    else:
        print(f"  ✗ FAILED: Expected 2 shifts remaining, got {len(filtered)}")
    
    print()


def test_mapping_fallbacks():
    """Test the three-tier mapping fallback logic."""
    print("Testing mapping fallback logic...")
    
    # Mock roster mapping
    roster_map = {
        'home': {
            12: 8471675,  # Jersey 12 -> Player ID 8471675
            23: 8474564,
        },
        'away': {
            12: 8475798,  # Same jersey number, different player
            45: 8476123,
        }
    }
    
    # Mock name mapping
    name_map = {
        'john smith': 8471675,
        'jane doe': 8474564,
        'bob jones': 8475798,
    }
    
    # Mock team IDs
    team_ids = {
        'home': 1,
        'away': 6,
    }
    
    # Test cases
    test_shifts = [
        {
            'desc': 'Jersey match on home team',
            'shift': {'player_number': 12, 'player_name': 'Smith, John', 'team_side': 'home'},
            'expected_id': 8471675,
            'expected_team_id': 1,
        },
        {
            'desc': 'Jersey match on away team (same number as home)',
            'shift': {'player_number': 12, 'player_name': 'Jones, Bob', 'team_side': 'away'},
            'expected_id': 8475798,
            'expected_team_id': 6,
        },
        {
            'desc': 'No jersey match, but name match',
            'shift': {'player_number': 99, 'player_name': 'Jane Doe', 'team_side': 'home'},
            'expected_id': 8474564,
            'expected_team_id': 1,
        },
        {
            'desc': 'Wrong team_side, should find in other team and correct',
            'shift': {'player_number': 45, 'player_name': None, 'team_side': 'home'},
            'expected_id': 8476123,
            'expected_team_id': 6,  # Should be corrected to away
            'expected_team_side': 'away',
        },
    ]
    
    for test in test_shifts:
        shift = test['shift'].copy()
        team_side = shift.get('team_side')
        player_number = shift.get('player_number')
        player_name = shift.get('player_name')
        
        # Apply mapping logic (simplified version of what's in get_shifts_from_nhl_html)
        canonical_id = None
        
        # First: Try jersey number mapping for detected team_side
        if player_number is not None and team_side in ('home', 'away'):
            team_roster = roster_map.get(team_side, {})
            canonical_id = team_roster.get(player_number)
        
        # Second: Try name-based mapping if jersey mapping failed
        if canonical_id is None and player_name:
            normalized = nhl_api._normalize_name(player_name)
            canonical_id = name_map.get(normalized)
        
        # Third: Try other team's roster
        if canonical_id is None and player_number is not None:
            other_side = 'away' if team_side == 'home' else 'home'
            other_roster = roster_map.get(other_side, {})
            alt = other_roster.get(player_number)
            if alt is not None:
                canonical_id = alt
                shift['team_side'] = other_side
                other_team_id = team_ids.get(other_side)
                if other_team_id is not None:
                    shift['team_id'] = other_team_id
        
        # Set canonical ID
        if canonical_id is not None:
            shift['player_id'] = int(canonical_id)
            
            # Infer team_id if not set
            if not shift.get('team_id'):
                if canonical_id in roster_map.get('home', {}).values():
                    shift['team_id'] = team_ids.get('home')
                elif canonical_id in roster_map.get('away', {}).values():
                    shift['team_id'] = team_ids.get('away')
        
        # Verify
        expected_id = test['expected_id']
        expected_team_id = test['expected_team_id']
        actual_id = shift.get('player_id')
        actual_team_id = shift.get('team_id')
        
        id_match = actual_id == expected_id
        team_match = actual_team_id == expected_team_id
        
        # Check team_side correction if expected
        team_side_match = True
        if 'expected_team_side' in test:
            team_side_match = shift.get('team_side') == test['expected_team_side']
        
        status = "✓" if (id_match and team_match and team_side_match) else "✗"
        print(f"  {status} {test['desc']}")
        if not id_match:
            print(f"     Player ID: expected {expected_id}, got {actual_id}")
        if not team_match:
            print(f"     Team ID: expected {expected_team_id}, got {actual_team_id}")
        if not team_side_match:
            print(f"     Team side: expected {test['expected_team_side']}, got {shift.get('team_side')}")
    
    print()


def main():
    print("Running mapping logic tests (no network required)")
    print("=" * 70)
    print()
    
    test_normalize_name()
    test_zero_length_filtering()
    test_mapping_fallbacks()
    
    print("=" * 70)
    print("Tests complete!")


if __name__ == '__main__':
    main()
