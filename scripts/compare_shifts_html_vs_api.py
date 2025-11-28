"""Compare shifts from primary API (`get_shifts`) with HTML fallback

Usage:
    python scripts/compare_shifts_html_vs_api.py --game 2025020232

The script fetches both sources via `nhl_api.get_shifts` and
`nhl_api.get_shifts_from_nhl_html`, normalizes the results into simple
lists/dataframes, and prints a short summary and sample mismatches.

This is a development helper to speed debugging of the HTML parser.
"""

import argparse
import pprint
import sys
from typing import List, Dict, Any, Optional

import pandas as pd

import nhl_api


def to_df_from_all_shifts(all_shifts: List[Dict[str, Any]]) -> pd.DataFrame:
    """Normalize a list of shift dicts into a pandas DataFrame with common columns."""
    rows = []
    for s in all_shifts or []:
        rows.append({
            'player_id': s.get('player_id') if 'player_id' in s else s.get('playerId') or s.get('player_id'),
            'player_number': s.get('player_number') or s.get('jersey') or s.get('playerNumber'),
            'team_id': s.get('team_id') or s.get('teamId') or s.get('team_id'),
            'team_side': s.get('team_side') or s.get('team') or s.get('team_side'),
            'period': s.get('period'),
            'start_seconds': s.get('start_seconds') or s.get('startTime') or s.get('start'),
            'end_seconds': s.get('end_seconds') or s.get('endTime') or s.get('end'),
            'start_raw': s.get('start_raw'),
            'end_raw': s.get('end_raw'),
            'raw': s,
        })
    df = pd.DataFrame(rows)
    return df


def overlap_fraction(a_start: Optional[float], a_end: Optional[float], b_start: Optional[float], b_end: Optional[float]) -> float:
    """Return fraction of overlap relative to the shorter interval. If either end is None, return 0."""
    try:
        if a_start is None or a_end is None or b_start is None or b_end is None:
            return 0.0
        a0, a1 = float(a_start), float(a_end)
        b0, b1 = float(b_start), float(b_end)
        # ensure ordering
        if a1 < a0:
            a0, a1 = a1, a0
        if b1 < b0:
            b0, b1 = b1, b0
        inter0 = max(a0, b0)
        inter1 = min(a1, b1)
        if inter1 <= inter0:
            return 0.0
        inter_len = inter1 - inter0
        shorter = min(a1 - a0, b1 - b0)
        if shorter <= 0:
            return 0.0
        return inter_len / shorter
    except Exception:
        return 0.0


def find_matches(html_df: pd.DataFrame, api_df: pd.DataFrame) -> Dict[str, Any]:
    """Try to match each html-derived shift to an api-derived shift.

    Matching priority:
    - exact player_id match (if player_id present in api) and overlap > 0.2
    - jersey/player_number match and overlap > 0.2

    Returns a dict with matched/unmatched lists and counts.
    """
    matches = []
    unmatched_html = []
    unmatched_api_idx = set(api_df.index.tolist())

    for hidx, h in html_df.iterrows():
        best_score = 0.0
        best_aidx = None
        for aidx, a in api_df.iterrows():
            score = 0.0
            # player id exact
            if pd.notna(a.player_id) and pd.notna(h.player_id) and str(a.player_id) == str(h.player_id):
                score += 1.0
            # player number match
            if pd.notna(a.player_number) and pd.notna(h.player_number) and str(a.player_number) == str(h.player_number):
                score += 0.6
            # same team side if available
            if pd.notna(a.team_side) and pd.notna(h.team_side) and str(a.team_side).lower() == str(h.team_side).lower():
                score += 0.1
            # time overlap fraction
            ofrac = overlap_fraction(h.start_seconds, h.end_seconds, a.start_seconds, a.end_seconds)
            score += ofrac
            if score > best_score:
                best_score = score
                best_aidx = aidx
        # accept match when score reasonably high
        if best_score >= 0.25 and best_aidx is not None:
            matches.append((hidx, best_aidx, best_score))
            if best_aidx in unmatched_api_idx:
                unmatched_api_idx.remove(best_aidx)
        else:
            unmatched_html.append(hidx)

    unmatched_api = list(unmatched_api_idx)
    return {'matches': matches, 'unmatched_html': unmatched_html, 'unmatched_api': unmatched_api}


def report(game_id: int):
    print('Fetching API shifts...')
    api_res = nhl_api.get_shifts(game_id)
    print('Fetching HTML shifts fallback...')
    html_res = nhl_api.get_shifts_from_nhl_html(game_id, debug=True)

    api_all = api_res.get('all_shifts') if isinstance(api_res, dict) else []
    html_all = html_res.get('all_shifts') if isinstance(html_res, dict) else []

    api_df = to_df_from_all_shifts(api_all)
    html_df = to_df_from_all_shifts(html_all)

    print('\nSummary:')
    print(' API shifts: total=', len(api_df), ' players=', len(api_df.player_id.dropna().unique()))
    print(' HTML shifts: total=', len(html_df), ' players=', len(html_df.player_number.dropna().unique()))

    match_info = find_matches(html_df, api_df)
    print('\nMatches found:', len(match_info['matches']))
    print('Unmatched html shifts:', len(match_info['unmatched_html']))
    print('Unmatched api shifts:', len(match_info['unmatched_api']))

    # Print a few unmatched examples
    def show_rows(df, idxs, name, n=5):
        print(f"\n{name} sample (up to {n}):")
        for i in idxs[:n]:
            row = df.loc[i].to_dict()
            pprint.pprint(row)

    show_rows(html_df, match_info['unmatched_html'], 'Unmatched HTML shifts')
    show_rows(api_df, match_info['unmatched_api'], 'Unmatched API shifts')

    # Print a few matched pairs with scores
    print('\nSample matches:')
    for (hidx, aidx, score) in match_info['matches'][:10]:
        print(f' HTML#{hidx} <-> API#{aidx} score={score:.2f}')
        pprint.pprint({'html': html_df.loc[hidx].to_dict(), 'api': api_df.loc[aidx].to_dict()})


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', '-g', type=int, required=True, help='Game ID to compare')
    args = parser.parse_args(argv)
    try:
        report(args.game)
    except Exception as e:
        print('Error while comparing shifts:', e)
        raise


if __name__ == '__main__':
    sys.exit(main())

