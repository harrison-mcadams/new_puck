#!/usr/bin/env python3
"""Test the /replot handler to ensure the game id received is the hidden 'game' field.

This script injects stub modules for `fit_xgs` and `plot` to avoid heavy network or plotting.
It posts various form payloads and prints what value the server passed to analyze_game.
"""
import sys
import types
import pandas as pd
from pathlib import Path

# ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import the Flask app
import app as appmod
flask_app = appmod.app

# Prepare a stub analyze_game that records the incoming game_id
records = []

def stub_analyze_game(game_id):
    print('stub_analyze_game called with:', repr(game_id))
    records.append(game_id)
    # return a tiny DataFrame so plot.plot_events is invoked but trivial
    return pd.DataFrame([{'event_type':'shot-on-goal','x_a':0,'y_a':0}])

# Prepare a stub plot.plot_events that writes a tiny file
def stub_plot_events(df, events_to_plot=None, out_path=None):
    print('stub_plot_events called; out_path=', out_path)
    try:
        with open(out_path, 'wb') as fh:
            fh.write(b'')
    except Exception as e:
        print('failed to write out_path:', e)

# Inject stubs into sys.modules so app's imports use them
sys.modules['fit_xgs'] = types.SimpleNamespace(analyze_game=stub_analyze_game)
sys.modules['plot'] = types.SimpleNamespace(plot_events=stub_plot_events)

# Test cases: each is data dict to POST
cases = [
    ({'game': '2025020196'}, 'hidden-only numeric'),
    ({'game': ' 2025020197 '}, 'hidden with whitespace'),
    ({'game': 'ABC-123'}, 'hidden non-numeric'),
    ({'game': '', 'game_manual': '2025020198'}, 'hidden empty + manual (client should set hidden, server will see empty)'),
    ({'game': '2025000000', 'game_manual': '2025020199'}, 'mismatch hidden vs manual (server uses hidden)'),
]

client = flask_app.test_client()

for data, desc in cases:
    print('\n--- POST', desc, 'payload=', data)
    resp = client.post('/replot', data=data, follow_redirects=False)
    print('Response status:', resp.status_code)
    if records:
        print('Last recorded analyze_game arg:', records[-1])
    else:
        print('No analyze_game call recorded')

print('\nAll records:', records)
