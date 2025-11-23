"""Test to reproduce the issue where team_id == 1 rows have player_id as jersey numbers."""
import os
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


def test_both_teams_get_player_id_mapped():
    """Test that both home and away teams get player_id properly mapped (not jersey numbers)."""
    fp = os.path.join(FIXTURE_DIR, 'shift_sample.html')
    orig_get = nhl_api.SESSION.get
    orig_get_game_feed = nhl_api.get_game_feed
    
    try:
        # Mock SESSION.get to read HTML fixture
        def mock_get(url, timeout=10):
            with open(fp, 'r', encoding='utf-8') as fh:
                return DummyResponse(fh.read(), status_code=200)
        nhl_api.SESSION.get = mock_get
        
        # Mock get_game_feed to return BOTH team rosters
        # This simulates a real game feed with rosterSpots for both teams
        def mock_get_game_feed(game_id):
            return {
                'homeTeam': {'id': 1, 'abbrev': 'BOS'},
                'awayTeam': {'id': 6, 'abbrev': 'NYR'},
                'rosterSpots': [
                    # Home team players (team_id = 1)
                    {'teamId': 1, 'sweaterNumber': 12, 'playerId': 8471675},
                    {'teamId': 1, 'sweaterNumber': 23, 'playerId': 8474564},
                    # Away team players (team_id = 6)
                    {'teamId': 6, 'sweaterNumber': 12, 'playerId': 8475798},
                    {'teamId': 6, 'sweaterNumber': 23, 'playerId': 8476453},
                ]
            }
        nhl_api.get_game_feed = mock_get_game_feed
        
        res = nhl_api.get_shifts_from_nhl_html(9999999999, force_refresh=True, debug=True)
        
        # Check roster mapping was successful
        assert res['debug']['roster_mapping']['home_players'] > 0, "Should have home players in roster"
        assert res['debug']['roster_mapping']['away_players'] > 0, "Should have away players in roster"
        
        # Check that all shifts have player_id mapped (not jersey numbers)
        for shift in res['all_shifts']:
            player_id = shift.get('player_id')
            player_number = shift.get('player_number')
            team_id = shift.get('team_id')
            
            # player_id should be a large number (NHL player IDs are typically 7-8 digits)
            # NOT the jersey number (which is typically 1-99)
            if player_number is not None and player_id is not None:
                assert player_id != player_number or player_id >= 1000, \
                    f"player_id {player_id} looks like jersey number {player_number} (team_id={team_id})"
        
        print(f"✓ Test passed: all {len(res['all_shifts'])} shifts have canonical player_id")
        
    finally:
        nhl_api.SESSION.get = orig_get
        nhl_api.get_game_feed = orig_get_game_feed


if __name__ == '__main__':
    test_both_teams_get_player_id_mapped()
    print("All tests passed!")
