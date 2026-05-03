"""
COMPREHENSIVE BLOCKED SHOT ATTRIBUTION AUDIT
=============================================
Tests multiple games from each modern era season.
Checks raw API, parse.py output, and post-pipeline CSV data.
Determines exactly which features are corrupted vs. which are unaffected.
"""
import requests
import pandas as pd
import numpy as np
import os
import sys
import random
import json

sys.path.insert(0, os.getcwd())

# ============================================================
# PART 1: Cross-Season API Verification
# ============================================================
print("=" * 80)
print("PART 1: CROSS-SEASON API VERIFICATION")
print("       Testing multiple games from each season")
print("=" * 80)

# Sample game IDs from each season (mix of regular + playoff)
season_games = {
    "20202021": [2020020001, 2020020100, 2020020500, 2020020800, 2020030111],
    "20212022": [2021020001, 2021020200, 2021020600, 2021020900, 2021030111],
    "20222023": [2022020001, 2022020300, 2022020700, 2022020999, 2022030111],
    "20232024": [2023020001, 2023020400, 2023020800, 2023021100, 2023030111],
    "20242025": [2024020001, 2024020300, 2024020700, 2024021000, 2024030111],
    "20252026": [2025020001, 2025020100, 2025020500, 2025020900, 2025021224],
}

api_results = {}
for season, game_ids in season_games.items():
    season_shooter_match = 0
    season_blocker_match = 0
    season_total = 0
    games_checked = 0
    
    for gid in game_ids:
        try:
            r = requests.get(
                f"https://api-web.nhle.com/v1/gamecenter/{gid}/play-by-play",
                timeout=5
            )
            if r.status_code != 200:
                continue
            data = r.json()
            
            home_id = data.get("homeTeam", {}).get("id")
            away_id = data.get("awayTeam", {}).get("id")
            if not home_id or not away_id:
                continue
            
            # Build roster map
            roster = data.get("rosterSpots", [])
            p_teams = {}
            for p in roster:
                pid = p.get("playerId")
                tid = p.get("teamId")
                if pid and tid:
                    p_teams[pid] = tid
            
            plays = [p for p in data.get("plays", []) if p.get("typeDescKey") == "blocked-shot"]
            
            for p in plays:
                d = p.get("details", {})
                owner_tid = d.get("eventOwnerTeamId")
                shooting_pid = d.get("shootingPlayerId")
                blocking_pid = d.get("blockingPlayerId")
                
                season_total += 1
                
                if shooting_pid and shooting_pid in p_teams:
                    if p_teams[shooting_pid] == owner_tid:
                        season_shooter_match += 1
                
                if blocking_pid and blocking_pid in p_teams:
                    if p_teams[blocking_pid] == owner_tid:
                        season_blocker_match += 1
            
            games_checked += 1
        except Exception as e:
            continue
    
    pct_shooter = (100 * season_shooter_match / season_total) if season_total > 0 else 0
    pct_blocker = (100 * season_blocker_match / season_total) if season_total > 0 else 0
    
    verdict = "SHOOTER" if pct_shooter > pct_blocker else "BLOCKER" if pct_blocker > pct_shooter else "INCONCLUSIVE"
    
    api_results[season] = {
        "games": games_checked,
        "total_blocks": season_total,
        "shooter_match": season_shooter_match,
        "blocker_match": season_blocker_match,
        "verdict": verdict
    }
    
    print(f"\n  {season}: {games_checked} games, {season_total} blocked shots")
    print(f"    Shooter == eventOwner: {season_shooter_match}/{season_total} ({pct_shooter:.1f}%)")
    print(f"    Blocker == eventOwner: {season_blocker_match}/{season_total} ({pct_blocker:.1f}%)")
    print(f"    --> eventOwnerTeamId is the {verdict}'S team")


# ============================================================
# PART 2: Check parse.py pathways  
# ============================================================
print("\n" + "=" * 80)
print("PART 2: PARSE.PY PATHWAYS")
print("       Checking all data gathering methods")
print("=" * 80)

from puck import parse

# Test with a known game
r = requests.get("https://api-web.nhle.com/v1/gamecenter/2025021224/play-by-play")
feed = r.json()

home_id = feed["homeTeam"]["id"]
away_id = feed["awayTeam"]["id"]
home_abb = feed["homeTeam"]["abbrev"]
away_abb = feed["awayTeam"]["abbrev"]

# Build ground truth from API
roster = feed.get("rosterSpots", [])
truth = {}
for p in roster:
    pid = p.get("playerId")
    tid = p.get("teamId")
    if pid and tid:
        truth[pid] = tid

print(f"\nTest game: {home_abb} (ID={home_id}) vs {away_abb} (ID={away_id})")

# Path A: parse._game() 
df_game = parse._game(feed)
bs_game = df_game[df_game["event"] == "blocked-shot"]
print(f"\nPath A: parse._game() -> {len(bs_game)} blocked shots")

correct_a = 0
for _, row in bs_game.iterrows():
    pid = row.get("player_id")
    tid = row.get("team_id")
    if pid in truth:
        if truth[pid] == tid:
            correct_a += 1
# Note: after our parse.py changes, this may be wrong now
print(f"  Player team == team_id: {correct_a}/{len(bs_game)}")
if correct_a == len(bs_game):
    print("  -> parse._game() attribution: CORRECT (shooter's team)")
else:
    print(f"  -> parse._game() attribution: MIXED/WRONG ({len(bs_game)-correct_a} mismatches)")
    # Show mismatches
    for _, row in bs_game.head(3).iterrows():
        pid = row.get("player_id")
        tid = row.get("team_id")
        pteam = truth.get(pid, "?")
        pname = row.get("player_name", "?")
        print(f"     {pname}: player_team={pteam}, row_team_id={tid}, match={pteam==tid}")

# Path B: parse._elaborate()
df_elab = parse._elaborate(df_game)
bs_elab = df_elab[df_elab["event"] == "blocked-shot"]
print(f"\nPath B: parse._elaborate() -> {len(bs_elab)} blocked shots")

correct_b = 0
for _, row in bs_elab.iterrows():
    pid = row.get("player_id")
    tid = row.get("team_id")
    if pid in truth:
        if truth[pid] == tid:
            correct_b += 1
print(f"  Player team == team_id: {correct_b}/{len(bs_elab)}")

# Check distance at this stage
if "distance" in bs_elab.columns:
    bs_dist = bs_elab["distance"].dropna()
    sog_dist = df_elab[df_elab["event"] == "shot-on-goal"]["distance"].dropna()
    print(f"  BS  avg distance: {bs_dist.mean():.1f}")
    print(f"  SOG avg distance: {sog_dist.mean():.1f}")
    if bs_dist.mean() > 100:
        print("  -> Distances look WRONG (measuring from wrong goal)")
    else:
        print("  -> Distances look CORRECT")


# ============================================================
# PART 3: Trace exactly what correction.py does
# ============================================================
print("\n" + "=" * 80)
print("PART 3: CORRECTION.PY IMPACT ANALYSIS")
print("       What features does the swap ACTUALLY affect?")
print("=" * 80)

from puck import correction

# Start from a clean _game() output (before any corrections)
# First, revert our parse.py changes effect by re-checking
df_clean = parse._game(feed)
bs_clean = df_clean[df_clean["event"] == "blocked-shot"].copy()

print(f"\nBefore correction.fix_blocked_shot_attribution():")
print(f"  Blocked shots: {len(bs_clean)}")

# Apply correction
df_corrected = correction.fix_blocked_shot_attribution(df_clean.copy())
bs_corrected = df_corrected[df_corrected["event"] == "blocked-shot"].copy()

print(f"\nAfter correction.fix_blocked_shot_attribution():")

# Check which columns changed
changed_cols = []
unchanged_cols = []
for col in bs_clean.columns:
    if col in bs_corrected.columns:
        try:
            before = bs_clean[col].astype(str).values
            after = bs_corrected[col].astype(str).values
            if not np.array_equal(before, after):
                changed_cols.append(col)
            else:
                unchanged_cols.append(col)
        except:
            unchanged_cols.append(col)

print(f"\n  CHANGED columns ({len(changed_cols)}): {changed_cols}")
print(f"  UNCHANGED columns ({len(unchanged_cols)}): {unchanged_cols[:20]}...")

# Specifically check what the model actually uses as features
model_features = [
    "distance", "angle_deg", "is_rebound", "rebound_angle_change", 
    "rebound_time_diff", "is_rush", "speed_from_last_event",
    "dist_from_last_event", "last_event_time_diff", "last_event_type",
    "score_diff", "game_state", "is_net_empty", "shot_type",
    "shooter_role", "time_elapsed_in_period_s", "total_time_elapsed_s",
    "is_home", "relative_game_state"
]

print(f"\n  Model feature impact:")
for feat in model_features:
    status = "CHANGED" if feat in changed_cols else "unchanged"
    print(f"    {feat}: {status}")


# ============================================================
# PART 4: What does the training pipeline actually see?
# ============================================================
print("\n" + "=" * 80)
print("PART 4: TRAINING PIPELINE FEATURE ANALYSIS")
print("       Checking what features are in the CSV and which ones are affected")
print("=" * 80)

# Load the actual training features list
try:
    from puck import config
    feat_list = getattr(config, "XGBOOST_FEATURES", None)
    if feat_list:
        print(f"\n  Model uses {len(feat_list)} features:")
        for f in feat_list:
            in_changed = f in changed_cols
            print(f"    {'* ' if in_changed else '  '}{f} {'<-- AFFECTED BY SWAP' if in_changed else ''}")
    else:
        print("  Could not load XGBOOST_FEATURES from config")
except Exception as e:
    print(f"  Error loading config: {e}")

# Check is_home derivation
print("\n  CRITICAL CHECK: How is 'is_home' derived?")
print("  If is_home = (team_id == home_id), then the swap FLIPS is_home for blocked shots")
print("  If is_home is derived from player roster data, it's unaffected")

# Check a specific CSV to see if is_home correlates with the swap
for season in ["20232024"]:
    csv_path = f"data/{season}/{season}_df.csv"
    if not os.path.exists(csv_path):
        continue
    sdf = pd.read_csv(csv_path, low_memory=False, nrows=5000)
    bs = sdf[sdf["event"] == "blocked-shot"]
    sog = sdf[sdf["event"] == "shot-on-goal"]
    
    if "is_home" in bs.columns:
        bs_home_pct = bs["is_home"].mean() * 100
        sog_home_pct = sog["is_home"].mean() * 100
        print(f"\n  {season} CSV: is_home % for blocked shots: {bs_home_pct:.1f}% (expect ~50%)")
        print(f"  {season} CSV: is_home % for shot-on-goal:   {sog_home_pct:.1f}% (expect ~51%)")
    
    if "distance" in bs.columns:
        # Distribution analysis
        bs_near = (bs["distance"] < 60).sum()
        bs_far = (bs["distance"] >= 60).sum()
        print(f"\n  Distance distribution for blocked shots:")
        print(f"    Near (<60ft): {bs_near}")
        print(f"    Far (>=60ft): {bs_far}")
        print(f"    Pct far: {100*bs_far/(bs_near+bs_far):.1f}%")

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
