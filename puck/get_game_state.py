# Ensure repository root is on sys.path so local module imports work when
# running this script directly from the `scripts/` directory.
import os, sys
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

def get_game_state(game_id, condition=None, return_df=False, return_per_game=False, df_shifts=None):
    """Return on-ice skater-count intervals for a single game using shift data.

    Small additions:
      - Consolidates contiguous/overlapping intervals with the same label.
      - Optional per-game summary output (when `return_per_game=True`) that
        mirrors the (intervals, cond_seconds, total_seconds) structure used
        by `timing.intervals_for_condition`.
      - Optional `df_shifts` argument to avoid redundant fetching/parsing.
    """
    import logging
    from collections import defaultdict
    import pandas as _pd
    try:
        from . import nhl_api
        from . import parse as _parse
    except Exception:
        logging.exception('Failed to import nhl_api or parse modules')
        return (None, None) if return_df else ([], [])

    # Defensive: normalize game_id
    try:
        gid = int(game_id)
    except Exception:
        try:
            gid = int(str(game_id).strip())
        except Exception:
            logging.error('Invalid game_id: %s', game_id)
            return (None, None) if return_df else ([], [])

    # Fetch shifts if not provided
    if df_shifts is None:
        shifts_res = nhl_api.get_shifts(gid)
        if not shifts_res or not isinstance(shifts_res, dict):
            logging.warning('get_game_state: no shifts for game %s', gid)
            return (None, None) if return_df else ([], [])

        # Convert to parsed shift rows (total seconds when available)
        df_shifts = _parse._shifts(shifts_res)
        if df_shifts is None or df_shifts.empty:
            logging.warning('get_game_state: parsed shifts empty for game %s; attempting force-refresh of shifts', gid)
            try:
                shifts_res2 = nhl_api.get_shifts(gid, force_refresh=True)
                df_shifts = _parse._shifts(shifts_res2)
                if df_shifts is None or df_shifts.empty:
                    # Provide richer diagnostics for debugging API/format issues
                    try:
                        summary = []
                        if isinstance(shifts_res2, dict):
                            if 'raw' in shifts_res2 and isinstance(shifts_res2['raw'], dict):
                                summary = list(shifts_res2['raw'].keys())[:20]
                            else:
                                summary = list(shifts_res2.keys())[:20]
                        else:
                            summary = [str(type(shifts_res2))]
                    except Exception:
                        summary = ['(failed to introspect shifts_res2)']
                    logging.warning('get_game_state: parsed shifts still empty after force_refresh for game %s; shifts_res2 keys/sample=%s', gid, summary)
                    return (None, None) if return_df else ([], [])
            except Exception:
                logging.exception('get_game_state: force-refresh get_shifts failed for game %s', gid)
                return (None, None) if return_df else ([], [])
    # Ensure we have start/end totals
    df_shifts = df_shifts.dropna(subset=['start_total_seconds', 'end_total_seconds'])
    if df_shifts.empty:
        logging.warning('get_game_state: no shifts with total-second bounds for %s', gid)
        return (None, None) if return_df else ([], [])

    # Fetch feed metadata to map home/away ids and abbs
    feed = nhl_api.get_game_feed(gid) or {}
    home_id = None; away_id = None; home_abb = None; away_abb = None
    try:
        if isinstance(feed, dict):
            h = feed.get('homeTeam') or feed.get('home') or {}
            a = feed.get('awayTeam') or feed.get('away') or {}
            if isinstance(h, dict):
                home_id = h.get('id') or h.get('teamId')
                home_abb = h.get('abbrev') or h.get('triCode') or h.get('name')
            if isinstance(a, dict):
                away_id = a.get('id') or a.get('teamId')
                away_abb = a.get('abbrev') or a.get('triCode') or a.get('name')
    except Exception:
        pass

    # Build event list for sweep-line: (time, type, team_id, player_id)
    events = []
    for _, r in df_shifts.iterrows():
        s = r.get('start_total_seconds')
        e = r.get('end_total_seconds')
        tid = r.get('team_id')
        pid = r.get('player_id')
        if s is None or e is None:
            continue
        try:
            si = int(s); ei = int(e)
        except Exception:
            continue
        if ei <= si:
            continue
        # Use type ordering so that 'end' processed before 'start' at same timestamp
        events.append((si, 'start', tid, pid))
        events.append((ei, 'end', tid, pid))

    if not events:
        logging.warning('get_game_state: no valid shift events for %s', gid)
        return (None, None) if return_df else ([], [])

    # sort by time then type (end before start)
    events.sort(key=lambda x: (x[0], 0 if x[1] == 'end' else 1))

    active = defaultdict(set)  # team_id -> set(player_id)
    intervals = []
    prev_t = events[0][0]

    # Helper to compute skaters from active players set
    def skaters_from_active(sset):
        # heuristic: assume one goalie when count>=1 -> skaters = max(0, n-1)
        n = len(sset)
        if n == 0:
            return 0
        return max(0, n - 1)

    i = 0
    n_events = len(events)
    while i < n_events:
        t, typ, tid, pid = events[i]
        # process all events at this timestamp together
        same_time = t
        # Before applying events at time `t`, the interval from prev_t -> t
        # reflects the current `active` sets.
        if t > prev_t:
            # compute home/away counts using fetched IDs if available
            home_count = skaters_from_active(active.get(home_id) or set()) if home_id is not None else skaters_from_active(active.get(list(active.keys())[0]) if active else set())
            away_count = skaters_from_active(active.get(away_id) or set()) if away_id is not None else skaters_from_active(active.get(list(active.keys())[1]) if len(active.keys())>1 else set())
            intervals.append({
                'start': prev_t,
                'end': t,
                'home_id': home_id,
                'away_id': away_id,
                'home_abb': home_abb,
                'away_abb': away_abb,
                'home_count': int(home_count),
                'away_count': int(away_count),
                'label': f"{int(home_count)}v{int(away_count)}",
            })
            prev_t = t

        # apply all events at this exact time
        while i < n_events and events[i][0] == same_time:
            _, etype, etid, epid = events[i]
            key_tid = etid
            if etype == 'start':
                try:
                    active[key_tid].add(epid)
                except Exception:
                    pass
            else:
                try:
                    if epid in active.get(key_tid, set()):
                        active[key_tid].remove(epid)
                except Exception:
                    pass
            i += 1
    # no more events; nothing further to append (game end is not known here)

    # Convert to DataFrame and consolidate contiguous intervals with same label
    df_intervals = _pd.DataFrame.from_records(intervals) if intervals else _pd.DataFrame()

    def _merge_intervals(df_int: 'pd.DataFrame'):
        if df_int is None or df_int.empty:
            return df_int
        # coerce numeric and sort
        df2 = df_int.copy()
        try:
            df2['start'] = pd.to_numeric(df2['start'], errors='coerce')
            df2['end'] = pd.to_numeric(df2['end'], errors='coerce')
        except Exception:
            pass
        df2 = df2.sort_values(by=['label', 'start']).reset_index(drop=True)
        merged_rows = []
        epsilon = 1e-6
        cur = None
        for idx, row in df2.iterrows():
            r = row.to_dict()
            # skip rows with missing bounds
            if r.get('start') is None or r.get('end') is None:
                continue
            if cur is None:
                cur = r.copy()
                continue
            # if same label and overlapping/touching, merge
            if r.get('label') == cur.get('label') and r.get('start') <= (cur.get('end', 0) + epsilon):
                # extend end and keep other fields from cur (home/away ids/abbs assumed same)
                cur['end'] = max(float(cur.get('end', 0)), float(r.get('end', 0)))
            else:
                merged_rows.append(cur)
                cur = r.copy()
        if cur is not None:
            merged_rows.append(cur)
        if not merged_rows:
            return _pd.DataFrame()
        return _pd.DataFrame.from_records(merged_rows)

    df_intervals = _merge_intervals(df_intervals)

    # If a condition is provided, compute filtered intervals and per-game summary
    filtered = None
    per_game = None
    if isinstance(condition, dict) and 'game_state' in condition:
        wanted = condition.get('game_state')
        if isinstance(wanted, (list, tuple, set)):
            wanted_set = set([str(w).strip() for w in wanted])
        else:
            wanted_set = {str(wanted).strip()}
        if not df_intervals.empty:
            # rows matching any of the wanted labels
            df_filtered = df_intervals[df_intervals['label'].astype(str).isin(wanted_set)].copy()
        else:
            df_filtered = _pd.DataFrame()
        filtered = df_filtered if return_df else df_filtered.to_dict('records')

        # build per-game-like output mirroring timing.intervals_for_condition
        if not df_intervals.empty and len(df_intervals) >= 1:
            times = sorted(list(df_intervals['start'].astype(float).tolist() + df_intervals['end'].astype(float).tolist()))
            total_seconds = float(max(times) - min(times)) if len(times) >= 2 else 0.0
        else:
            total_seconds = 0.0
        cond_seconds = 0.0
        cond_intervals = []
        if not df_filtered.empty:
            for _, r in df_filtered.iterrows():
                s = float(r['start']); e = float(r['end'])
                cond_intervals.append((s, e))
                cond_seconds += (e - s)
        per_game = {str(gid): (cond_intervals, float(cond_seconds), float(total_seconds))}
    else:
        filtered = None

    # Return shape: keep backward compatible (intervals, filtered) when not requesting per_game
    if return_per_game:
        return (df_intervals if return_df else df_intervals.to_dict('records'), filtered, per_game)
    return (df_intervals, filtered) if return_df else (df_intervals.to_dict('records'), filtered)


# Simple CLI/demo
if __name__ == '__main__':
    import argparse, sys
    parser = argparse.ArgumentParser(description='Compute game_state intervals from shifts')
    parser.add_argument('--game_id', '-g', type=int, default=None, help='Game ID to inspect')
    parser.add_argument('--game_state', '-s', nargs='*', help="Optional game_state(s) to filter, e.g. '5v4' or '5v5'")
    args = parser.parse_args()
    if args.game_id is None:
        try:
            import nhl_api
            gid = nhl_api.get_game_id(team='PHI')
            print('No game_id provided; using most recent PHI game ->', gid)
        except Exception:
            print('No game_id provided and failed to discover recent game; exiting')
            sys.exit(1)
    else:
        gid = args.game_id
    cond = {'game_state': args.game_state} if args.game_state else None

    ints, filt = get_game_state(gid, condition=cond, return_df=True)
    print('\nAll intervals:')
    print(ints.head().to_string()) if hasattr(ints, 'head') else print(ints)
    if filt is not None:
        print('\nFiltered intervals:')
        print(filt.head().to_string()) if hasattr(filt, 'head') else print(filt)
