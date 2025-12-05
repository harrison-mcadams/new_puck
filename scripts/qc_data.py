"""Data quality checks for scraped NHL event data.

This script performs basic QC screening to find games/periods where the
recorded attacking direction may be wrong. The concrete check implemented
here is:

  - For each game / period / team, compute mean distance to the "attacked"
    goal based on the dataset's recorded defending side (the same logic used
    elsewhere in the codebase), and compare it to the mean distance to the
    nearest goal (ground truth independent of orientation). If the mean
    attacked distance is substantially larger than the nearest-goal mean
    (absolute or ratio-based threshold), the group is flagged as suspicious.

Usage:
    python3 scripts/qc_data.py --data-dir data --out reports/qc_report.json

Outputs:
  - prints a short summary to stdout
  - writes a JSON/CSV report with flagged groups and metrics to --out

The script is intentionally conservative: it requires at least
`min_shots` events in a group before flagging.

It also includes a list of suggested additional QC checks in the
`EXTRA_QC_PSEUDO` string for brainstorming next steps.

"""

from pathlib import Path
import argparse
import json
import pandas as pd
import numpy as np
import math
from typing import List, Dict, Any, Optional

# try to import rink helper for canonical goal x coordinates
# Add project root to sys.path to allow importing puck package
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from puck.rink import rink_goal_xs
except Exception:
    # fallback constants if rink helper not available
    def rink_goal_xs():
        return -89.0, 89.0

SHOT_TYPES = set(['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal'])

EXTRA_QC_PSEUDO = """
Other QC ideas (pseudo-code / brainstorming):

1) Coordinate bounds and rink mask:
   - Verify x,y fall inside rink bounds (use rink.rink_bounds / rink_half_height_at_x)
   - Flag events outside rink (possible parsing errors or coordinate swaps)

2) Sudden team-side flips mid-period:
   - For each game, compute majority attacked side for each team per period.
   - If a team flips attacked side within the same period, flag (rare / suspicious).

3) Angle/distribution checks:
   - Compute angle_deg distributions for each team; if angles cluster around 180° offset
     relative to expected, there may be a flip.

4) Time continuity / duplicates:
   - Find duplicate events (same game_id, period, time, coords) -> likely duplicate scraping.

5) Event type consistency:
   - If `goal` events have x/y missing or far outside rink, flag.

6) Team identity mismatch:
   - Use player/team IDs to verify player roster mapping for that game (mis-mapped team IDs possible).

7) Home/away handedness:
   - If a team's 'home_team_defending_side' is inconsistent across events in the same game, flag.

8) Aggregate stats sanity:
   - Compare per-game mean shot distance to historical league percentiles for that rink.
     Outliers (e.g., > 99.9th percentile) are suspicious.

9) Auto-correction heuristics (only after manual review):
   - If a group is flagged and flipping x->-x (or swapping left/right goal assignment)
     reduces mean distance dramatically, propose an automated correction.

10) Can consider explicit check against another NHL score repository. 
Specifically the NHL also collates play-by-play data with a more fleshed out 
HTML page. Here is an example call: 
https://www.nhl.com/scores/htmlreports/20142015/PL010078.HTM. The last two 
parts of the URL are a combination of the game_id

"""


def load_csv_candidates(data_dir: Path) -> List[Path]:
    """Return a list of CSV files to load from a data directory.

    - Preferred pattern: data/processed/{year}/{year}_df.csv
    - Also accept any .csv files found anywhere under the directory.
    """
    files: List[Path] = []

    # Normalize and resolve the requested path
    try:
        base = Path(data_dir).expanduser().resolve()
    except Exception:
        base = Path(data_dir)

    tried_dirs: List[Path] = []
    if base.exists() and base.is_dir():
        tried_dirs.append(base)
    else:
        # Common fallbacks
        for alt in (Path('data'), Path('static')):
            if alt.exists() and alt.is_dir():
                tried_dirs.append(alt.resolve())
                break

    # If we still have nothing, fall back to project root so the script at least scans the workspace
    if not tried_dirs:
        tried_dirs.append(Path('.').resolve())

    # First pass: look for per-year files like {year}/{year}_df.csv
    for d in tried_dirs:
        try:
            for child in sorted([p for p in d.iterdir() if p.is_dir()]):
                cand = child / f"{child.name}_df.csv"
                if cand.exists():
                    files.append(cand.resolve())
        except Exception:
            continue

    # Second pass: if none found, search recursively for any CSV under the first tried dir
    if not files:
        search_root = tried_dirs[0]
        try:
            files = [p.resolve() for p in search_root.rglob('*.csv')]
        except Exception:
            files = []

    # Deduplicate while preserving order
    out: List[Path] = []
    seen = set()
    for f in files:
        if f in seen:
            continue
        seen.add(f)
        out.append(f)

    if not out:
        print(f'load_csv_candidates: no CSV files found under {base} or fallbacks. Tried: {[str(p) for p in tried_dirs]}')

    return out


def compute_distances(df: pd.DataFrame) -> pd.DataFrame:
    """Add computed distance columns to the DataFrame.

    Columns added:
      - dist_left_goal
      - dist_right_goal
      - dist_nearest_goal
      - dist_attacked_goal (based on home_team_defending_side + team/home logic)
      - attacked_goal_side ("left" or "right") as inferred for the event

    The function is robust to missing columns and will attempt reasonable
    fallbacks.
    """
    left_x, right_x = rink_goal_xs()
    df = df.copy()

    # Ensure numeric x/y
    df['x'] = pd.to_numeric(df.get('x', pd.Series(np.nan)), errors='coerce')
    df['y'] = pd.to_numeric(df.get('y', pd.Series(np.nan)), errors='coerce')

    # distances to canonical goals
    df['dist_left_goal'] = np.hypot(df['x'] - left_x, df['y'] - 0.0)
    df['dist_right_goal'] = np.hypot(df['x'] - right_x, df['y'] - 0.0)
    df['dist_nearest_goal'] = df[['dist_left_goal', 'dist_right_goal']].min(axis=1)

    # infer attacked goal per event using same logic as _elaborate
    def infer_attacked_side(row) -> Optional[str]:
        # prefer explicit home_team_defending_side when available
        hside = row.get('home_team_defending_side') or row.get('homeTeamDefendingSide')
        try:
            team_id = int(row.get('team_id')) if row.get('team_id') is not None else None
        except Exception:
            team_id = None
        try:
            home_id = int(row.get('home_id')) if row.get('home_id') is not None else None
        except Exception:
            home_id = None

        if team_id is None:
            return None
        # if shooter is home
        if team_id == home_id:
            if hside == 'left':
                return 'right'
            elif hside == 'right':
                return 'left'
            else:
                return 'right'
        else:
            # shooter is away -> attacking goal is side not defended by home
            if hside == 'left':
                return 'left'
            elif hside == 'right':
                return 'right'
            else:
                return 'left'

    df['attacked_goal_side'] = df.apply(infer_attacked_side, axis=1)

    def attacked_dist(row):
        s = row.get('attacked_goal_side')
        if s == 'left':
            return row['dist_left_goal']
        elif s == 'right':
            return row['dist_right_goal']
        else:
            return np.nan

    df['dist_attacked_goal'] = df.apply(attacked_dist, axis=1)
    return df


def qc_flag_groups(df: pd.DataFrame, min_shots: int = 3, abs_thresh: float = 20.0, ratio_thresh: float = 1.25) -> pd.DataFrame:
    """Run the QC check and return a DataFrame of flagged groups.

    Grouping: [game_id, period, team_id]
    Flags when:
      - group_n >= min_shots AND
      - (mean_dist_attacked - mean_dist_nearest) >= abs_thresh OR
        mean_dist_attacked / mean_dist_nearest >= ratio_thresh
      - OR more than 50% of events in group have dist_attacked > dist_nearest

    Returns a DataFrame with metrics and a boolean 'flagged' column.
    """
    required = ['game_id', 'period', 'team_id', 'dist_attacked_goal', 'dist_nearest_goal', 'event']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in dataframe")

    # only shot attempts
    df_shots = df[df['event'].isin(SHOT_TYPES)].copy()

    grp = df_shots.groupby(['game_id', 'period', 'team_id'])

    stats = grp.agg(
        n_shots=('event', 'count'),
        mean_attacked=('dist_attacked_goal', 'mean'),
        mean_nearest=('dist_nearest_goal', 'mean'),
        median_attacked=('dist_attacked_goal', 'median'),
        median_nearest=('dist_nearest_goal', 'median'),
        prop_attacked_far=('dist_attacked_goal', lambda s: np.mean(s > df_shots.loc[s.index, 'dist_nearest_goal']))
    ).reset_index()

    # compute derived metrics
    stats['diff_mean'] = stats['mean_attacked'] - stats['mean_nearest']
    stats['ratio_mean'] = stats['mean_attacked'] / stats['mean_nearest']

    # flagging rules
    cond1 = (stats['n_shots'] >= min_shots) & (stats['diff_mean'] >= abs_thresh)
    cond2 = (stats['n_shots'] >= min_shots) & (stats['ratio_mean'] >= ratio_thresh)
    cond3 = (stats['n_shots'] >= min_shots) & (stats['prop_attacked_far'] > 0.5)

    stats['flagged'] = cond1 | cond2 | cond3

    # add human-readable reason
    reasons = []
    for i, r in stats.iterrows():
        rs = []
        if r['n_shots'] < min_shots:
            rs.append('too_few_shots')
        else:
            if r['diff_mean'] >= abs_thresh:
                rs.append(f'mean_diff_gt_{abs_thresh}')
            if r['ratio_mean'] >= ratio_thresh:
                rs.append(f'mean_ratio_gt_{ratio_thresh:.2f}')
            if r['prop_attacked_far'] > 0.5:
                rs.append('majority_attacked_farther')
        reasons.append(','.join(rs) if rs else '')
    stats['reason'] = reasons

    flagged = stats[stats['flagged']].copy()
    return stats, flagged


def run_qc(data_dir: Path, out: Optional[Path] = None, min_shots: int = 3, abs_thresh: float = 20.0, ratio_thresh: float = 1.25):
    files = load_csv_candidates(data_dir)
    if not files:
        print(f'No CSV candidates found under {data_dir}')
        return 1

    print('Found CSVs:')
    for f in files:
        print('  ', f)

    frames = []
    for f in files:
        try:
            d = pd.read_csv(f)
            d['_source_file'] = str(f)
            frames.append(d)
        except Exception as e:
            print('Failed to read', f, e)

    if not frames:
        print('No frames loaded')
        return 1

    df = pd.concat(frames, ignore_index=True)
    print('Loaded rows:', len(df))

    # ensure columns exist; try common alternate column names
    col_map = {}
    # normalize some column names if present with alternate names
    if 'gamePk' in df.columns and 'game_id' not in df.columns:
        df['game_id'] = df['gamePk']
    if 'game_id' not in df.columns and 'gamePk' not in df.columns:
        # try 'gameID'
        if 'gameID' in df.columns:
            df['game_id'] = df['gameID']
    if 'period' not in df.columns and 'periodNumber' in df.columns:
        df['period'] = df['periodNumber']

    # compute distances/attacked side
    df2 = compute_distances(df)

    # run group-based QC
    stats, flagged = qc_flag_groups(df2, min_shots=min_shots, abs_thresh=abs_thresh, ratio_thresh=ratio_thresh)

    print('\nQC summary:')
    print('  total groups checked:', len(stats))
    print('  flagged groups:', len(flagged))
    if not flagged.empty:
        print('\nTop flagged groups (first 20):')
        print(flagged[['game_id','period','team_id','n_shots','mean_attacked','mean_nearest','diff_mean','ratio_mean','prop_attacked_far','reason']].head(20).to_string(index=False))

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        # write both JSON and CSV summaries
        summary = {
            'checked_groups': len(stats),
            'flagged': len(flagged),
            'params': {'min_shots': min_shots, 'abs_thresh': abs_thresh, 'ratio_thresh': ratio_thresh}
        }
        with open(out, 'w', encoding='utf-8') as fh:
            json.dump({'summary': summary, 'flagged': flagged.to_dict('records')}, fh, indent=2)
        csv_out = out.with_suffix('.csv')
        stats.to_csv(csv_out, index=False)
        print('\nWrote report to', out, 'and', csv_out)

    # print extra QC pseudo-code to help brainstorm
    print('\nAdditional QC ideas (pseudo-code) - see EXTRA_QC_PSEUDO in the script for details')
    print(EXTRA_QC_PSEUDO)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QC scraped NHL event CSVs')
    parser.add_argument('--data-dir', default='../data', help='Root folder to '
                                                            'scan for per-year CSVs (default data/)')
    parser.add_argument('--out', default='../reports/qc_report.json',
                        type=Path, help='Output JSON path for flagged report')
    parser.add_argument('--min-shots', default=3, type=int, help='Minimum shots in a group to consider')
    parser.add_argument('--abs-thresh', default=20.0, type=float, help='Absolute distance (ft) difference threshold')
    parser.add_argument('--ratio-thresh', default=1.25, type=float, help='Ratio threshold for mean_attacked / mean_nearest')

    args = parser.parse_args()
    res = run_qc(Path(args.data_dir), out=args.out, min_shots=args.min_shots, abs_thresh=args.abs_thresh, ratio_thresh=args.ratio_thresh)
    raise SystemExit(res)
