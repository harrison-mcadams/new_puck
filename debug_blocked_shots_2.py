"""
Diagnostic Part 2: Verify that correction.py is DOUBLE-SWAPPING blocked shots.

We already know from Part 1 that the raw API has eventOwnerTeamId == Shooter's team.
The correction module assumes the opposite and swaps again.

This script:
1. Loads a raw game feed directly from the API
2. Parses it through _game() (parse.py) - check team_id assignment
3. Runs correction.fix_blocked_shot_attribution() - check if it BREAKS the assignment
4. Compares distance before/after correction
"""
import sys
import os
sys.path.insert(0, os.getcwd())

import requests
import pandas as pd
import numpy as np
from puck import parse, correction

game_id = 2025021224

# Step 1: Fetch raw feed
print("=" * 70)
print("STEP 1: Fetch raw game feed from API")
print("=" * 70)
r = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play")
feed = r.json()

home_id = feed["homeTeam"]["id"]
away_id = feed["awayTeam"]["id"]
home_abb = feed["homeTeam"]["abbrev"]
away_abb = feed["awayTeam"]["abbrev"]
print(f"Home: {home_abb} (ID={home_id}), Away: {away_abb} (ID={away_id})")

# Build roster for verification
roster = feed.get("rosterSpots", [])
player_info = {}
for p in roster:
    pid = p.get("playerId")
    tid = p.get("teamId")
    fn = p.get("firstName", {})
    ln = p.get("lastName", {})
    fname = fn.get("default") if isinstance(fn, dict) else fn
    lname = ln.get("default") if isinstance(ln, dict) else ln
    if pid:
        player_info[pid] = {
            "name": f"{fname} {lname}" if fname and lname else "Unknown",
            "team_id": tid,
            "team": home_abb if tid == home_id else away_abb
        }

# Step 2: Parse through _game()
print("\n" + "=" * 70)
print("STEP 2: Parse through _game() - check team_id on blocked shots")
print("=" * 70)
df = parse._game(feed)
blocked = df[df["event"] == "blocked-shot"].copy()
print(f"Found {len(blocked)} blocked-shot rows after _game()")

# Check team_id against player_info
print("\nFirst 5 blocked shots BEFORE correction:")
for idx, row in blocked.head(5).iterrows():
    pid = row.get("player_id")
    tid = row.get("team_id")
    pinfo = player_info.get(pid, {"name": "?", "team": "?", "team_id": None})
    
    # Is the team_id the shooter's team or the blocker's team?
    tid_team = home_abb if str(tid) == str(home_id) else away_abb
    player_actual_team = pinfo["team"]
    
    match = "CORRECT" if tid_team == player_actual_team else "WRONG"
    print(f"  Player: {pinfo['name']} ({player_actual_team}) | "
          f"team_id={tid} ({tid_team}) | "
          f"Assessment: {match}")

# Step 3: Run correction
print("\n" + "=" * 70)
print("STEP 3: Run correction.fix_blocked_shot_attribution()")
print("=" * 70)
df_corrected = correction.fix_blocked_shot_attribution(df.copy())
blocked_c = df_corrected[df_corrected["event"] == "blocked-shot"].copy()

print("First 5 blocked shots AFTER correction:")
for idx, row in blocked_c.head(5).iterrows():
    pid = row.get("player_id")
    tid = row.get("team_id")
    pinfo = player_info.get(pid, {"name": "?", "team": "?", "team_id": None})
    
    tid_team = home_abb if str(tid) == str(home_id) else (away_abb if str(tid) == str(away_id) else "?")
    player_actual_team = pinfo["team"]
    
    match = "CORRECT" if tid_team == player_actual_team else "WRONG (DOUBLE-SWAPPED!)"
    print(f"  Player: {pinfo['name']} ({player_actual_team}) | "
          f"team_id={tid} ({tid_team}) | "
          f"Assessment: {match}")

# Step 4: Distance comparison
print("\n" + "=" * 70)
print("STEP 4: Distance comparison")
print("=" * 70)
if "distance" in blocked.columns and "distance" in blocked_c.columns:
    d_before = blocked["distance"].dropna()
    d_after = blocked_c["distance"].dropna()
    print(f"  Before correction - mean: {d_before.mean():.1f}, median: {d_before.median():.1f}")
    print(f"  After correction  - mean: {d_after.mean():.1f}, median: {d_after.median():.1f}")
    print(f"  Shot-on-goal avg  - mean: {df[df['event']=='shot-on-goal']['distance'].dropna().mean():.1f}")
else:
    print("  Distance column not available at this stage (computed later in pipeline)")

# Step 5: Check what the training CSVs actually show
print("\n" + "=" * 70)
print("STEP 5: Training CSV sanity check - are distances reasonable?")
print("=" * 70)
print("(If blocked shot distances are ~155, the correction is DOUBLE-SWAPPING)")
print("(If blocked shot distances are ~35-45, the correction is CORRECT)")

for season in ["20202021", "20212022", "20222023", "20232024", "20242025"]:
    csv_path = f"data/{season}/season_{season}.csv"
    if not os.path.exists(csv_path):
        continue
    sdf = pd.read_csv(csv_path, low_memory=False)
    bs = sdf[sdf["event"] == "blocked-shot"]
    sog = sdf[sdf["event"] == "shot-on-goal"]
    
    bs_dist = bs["distance"].dropna().mean() if "distance" in bs.columns else float("nan")
    sog_dist = sog["distance"].dropna().mean() if "distance" in sog.columns else float("nan")
    
    verdict = "LIKELY BROKEN" if bs_dist > 100 else "OK"
    print(f"  {season}: BS avg_dist={bs_dist:.1f}, SOG avg_dist={sog_dist:.1f} -> {verdict}")
