import os
import pytest
from types import SimpleNamespace

import nhl_api

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')

class DummyResponse:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code
        self.headers = {}
    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f'Status {self.status_code}')
    def json(self):
        raise ValueError('No JSON')

def _patch_session_get(tmpfile_path, only_home=False):
    # return a function that mimics requests.Session.get
    def _get(url, timeout=10):
        # If only_home is True, only return content for TH (home) URLs
        if only_home and '/TV' in url:
            # Simulate a 404 for away report
            raise Exception('404 Not Found')
        with open(tmpfile_path, 'r', encoding='utf-8') as fh:
            return DummyResponse(fh.read(), status_code=200)
    return _get


def test_parse_sample_html(tmp_path):
    fp = os.path.join(FIXTURE_DIR, 'shift_sample.html')
    # patch SESSION.get to read local file (only for home report)
    orig_get = nhl_api.SESSION.get
    orig_get_game_feed = nhl_api.get_game_feed
    try:
        nhl_api.SESSION.get = _patch_session_get(fp, only_home=True)
        # Mock get_game_feed to return team info
        def mock_get_game_feed(game_id):
            return {
                'homeTeam': {'id': 1, 'abbrev': 'BOS'},
                'awayTeam': {'id': 6, 'abbrev': 'NYR'}
            }
        nhl_api.get_game_feed = mock_get_game_feed
        
        res = nhl_api.get_shifts_from_nhl_html(9999999999, force_refresh=True, debug=True)
        assert isinstance(res, dict)
        assert 'all_shifts' in res
        assert len(res['all_shifts']) == 2
        # check parsed fields
        first = res['all_shifts'][0]
        assert first['player_number'] == 12
        assert first['start_seconds'] == 5*60 + 12
        # verify team_id is set
        assert first['team_id'] == 1, "team_id should be set from game feed"
        assert first['team_side'] == 'home'
        # verify debug info
        assert res['debug']['team_id_set_count'] == 2, "team_id should be set for all shifts"
    finally:
        nhl_api.SESSION.get = orig_get
        nhl_api.get_game_feed = orig_get_game_feed


def test_summary_fixture_ignored(tmp_path):
    fp = os.path.join(FIXTURE_DIR, 'shift_edge_summary.html')
    orig_get = nhl_api.SESSION.get
    try:
        nhl_api.SESSION.get = _patch_session_get(fp)
        res = nhl_api.get_shifts_from_nhl_html(1234567890, force_refresh=True, debug=True)
        assert isinstance(res, dict)
        assert 'all_shifts' in res
        # summary-only tables should produce zero per-player shifts
        assert len(res['all_shifts']) == 0
    finally:
        nhl_api.SESSION.get = orig_get


def test_player_id_mapping_with_roster():
    """Test that player_id is mapped from jersey number to canonical ID when roster data is available."""
    fp = os.path.join(FIXTURE_DIR, 'shift_sample.html')
    orig_get = nhl_api.SESSION.get
    orig_get_game_feed = nhl_api.get_game_feed
    orig_get_roster = nhl_api._get_roster_mapping
    try:
        nhl_api.SESSION.get = _patch_session_get(fp, only_home=True)
        
        # Mock get_game_feed to return team info
        def mock_get_game_feed(game_id):
            return {
                'homeTeam': {'id': 1, 'abbrev': 'BOS'},
                'awayTeam': {'id': 6, 'abbrev': 'NYR'},
                'rosterSpots': [
                    {'teamId': 1, 'sweaterNumber': 12, 'playerId': 8471675, 'firstName': 'John', 'lastName': 'Smith'}
                ]
            }
        nhl_api.get_game_feed = mock_get_game_feed
        
        # Mock _get_roster_mapping to return roster with jersey -> player_id mapping
        def mock_get_roster_mapping(game_id):
            return {
                'home': {12: 8471675},  # jersey 12 maps to player_id 8471675
                'away': {}
            }
        nhl_api._get_roster_mapping = mock_get_roster_mapping
        
        res = nhl_api.get_shifts_from_nhl_html(9999999999, force_refresh=True, debug=True)
        
        assert isinstance(res, dict)
        assert len(res['all_shifts']) == 2
        
        # Verify both shifts have canonical player_id
        for shift in res['all_shifts']:
            assert shift['player_number'] == 12
            assert shift['player_id'] == 8471675, "player_id should be canonical (not jersey number)"
            assert shift['team_id'] == 1
            assert shift['team_side'] == 'home'
        
        # Verify debug info shows successful mapping
        assert res['debug']['roster_mapping']['mapped_shifts'] == 2
        assert res['debug']['roster_mapping']['unmapped_shifts'] == 0
        
    finally:
        nhl_api.SESSION.get = orig_get
        nhl_api.get_game_feed = orig_get_game_feed
        nhl_api._get_roster_mapping = orig_get_roster

