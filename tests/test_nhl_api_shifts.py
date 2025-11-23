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

def _patch_session_get(tmpfile_path):
    # return a function that mimics requests.Session.get
    def _get(url, timeout=10):
        with open(tmpfile_path, 'r', encoding='utf-8') as fh:
            return DummyResponse(fh.read(), status_code=200)
    return _get


def test_parse_sample_html(tmp_path):
    fp = os.path.join(FIXTURE_DIR, 'shift_sample.html')
    # patch SESSION.get to read local file
    orig_get = nhl_api.SESSION.get
    try:
        nhl_api.SESSION.get = _patch_session_get(fp)
        res = nhl_api.get_shifts_from_nhl_html(9999999999, force_refresh=True, debug=True)
        assert isinstance(res, dict)
        assert 'all_shifts' in res
        assert len(res['all_shifts']) == 2
        # check parsed fields
        first = res['all_shifts'][0]
        assert first['player_number'] == 12
        assert first['start_seconds'] == 5*60 + 12
    finally:
        nhl_api.SESSION.get = orig_get


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

