"""Test parity checks for HTML shift parsing."""
import pytest
import os
import sys
import pandas as pd

# Import from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import nhl_api
import parse

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')


class DummyResponse:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code
        self.headers = {}
    
    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f'Status {self.status_code}')


def test_html_shifts_parity_checks():
    """
    Programmatic parity checks for HTML shift parsing:
    1) No summary headers in HTML data
    2) Key columns present and non-null
    3) player_id numeric
    4) team_id set for all shifts
    """
    fp = os.path.join(FIXTURE_DIR, 'shift_sample.html')
    
    orig_get = nhl_api.SESSION.get
    orig_feed = nhl_api.get_game_feed
    
    try:
        # Mock SESSION.get
        def mock_get(url, timeout=10):
            # Only return for home URL (to avoid duplicate shifts)
            if '/TH' in url:
                with open(fp, 'r', encoding='utf-8') as fh:
                    return DummyResponse(fh.read(), status_code=200)
            else:
                # Simulate 404 for away
                raise Exception('404 Not Found')
        
        nhl_api.SESSION.get = mock_get
        
        # Mock game feed with roster for home team
        def mock_feed(gid):
            return {
                'homeTeam': {'id': 1, 'abbrev': 'BOS'},
                'awayTeam': {'id': 6, 'abbrev': 'NYR'},
                'rosterSpots': [
                    {'teamId': 1, 'sweaterNumber': 12, 'playerId': 8471675},
                ]
            }
        
        nhl_api.get_game_feed = mock_feed
        
        # Get shifts and parse
        html_res = nhl_api.get_shifts_from_nhl_html(999, force_refresh=True)
        html = parse._shifts(html_res)
        
        # Check 1: No summary headers in HTML data
        bad_tokens = ['Per', 'SHF', 'AVG', 'TOI', 'EV TOT', 'PP TOT', 'SH TOT']
        for t in bad_tokens:
            # Check each row to see if any cell contains the summary token
            for idx, row in html.iterrows():
                row_str = ' '.join(str(v) for v in row.values)
                assert t not in row_str.upper(), f"Found summary token '{t}' in row {idx}"
        
        # Check 2: Key columns present
        required_cols = ['team_id', 'player_id', 'start_raw', 'end_raw', 
                        'start_total_seconds', 'end_total_seconds']
        for col in required_cols:
            assert col in html.columns, f"Missing column: {col}"
        
        # Check 3: Key columns are non-null
        for col in ['team_id', 'start_raw', 'end_raw', 'start_total_seconds', 'end_total_seconds']:
            assert html[col].notnull().all(), f"Column {col} has null values"
        
        # player_id can be null if unmapped, but should be present in some rows
        assert 'player_id' in html.columns
        assert html['player_id'].notnull().any(), "No player_id values found"
        
        # Check 4: player_id is numeric where present
        player_ids = html['player_id'].dropna()
        numeric_ids = pd.to_numeric(player_ids, errors='coerce')
        assert numeric_ids.notnull().all(), "player_id contains non-numeric values"
        
        # Check 5: Mapped player_ids should be >= 1000 (real NHL IDs, not jersey numbers)
        mapped_ids = numeric_ids[numeric_ids >= 100]  # Filter to likely mapped IDs
        if len(mapped_ids) > 0:
            assert (mapped_ids >= 1000).all(), \
                f"Some player_id values look like jersey numbers: {mapped_ids[mapped_ids < 1000].tolist()}"
        
    finally:
        nhl_api.SESSION.get = orig_get
        nhl_api.get_game_feed = orig_feed


if __name__ == '__main__':
    test_html_shifts_parity_checks()
    print("\n✅ All parity checks passed!")
