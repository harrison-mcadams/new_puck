#!/usr/bin/env python3
"""Small smoke-run wrapper for compare_shifts that can run in offline mode.

Usage: OFFLINE=1 python scripts/smoke_compare_shifts.py --game 2025020232
"""
import os
import sys
import argparse
import json

import nhl_api

parser = argparse.ArgumentParser()
parser.add_argument('--game', type=int, required=True)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

if os.environ.get('OFFLINE'):
    print('Offline mode: will not call live APIs; compare_shifts will be run but network calls are patched if possible.')

try:
    out = nhl_api.compare_shifts(args.game, debug=args.debug)
    print('API count:', out['api']['count'])
    print('HTML count:', out['html']['count'])
    print('Only in API players:', out['diff']['only_api_players'])
    print('Only in HTML players:', out['diff']['only_html_players'])
except Exception as e:
    print('Error running compare_shifts:', e)
    sys.exit(2)

