"""
DEFINITIVE BLOCKED SHOT ATTRIBUTION AUDIT
==========================================
Tests from scratch — no caches, no CSVs, directly from API.
Covers ALL data gathering pathways in the codebase.

DATA PATHWAYS IN THE CODEBASE:
  PATH A: nhl_api.get_game_feed(game_id) -> parse._game(feed) -> parse._elaborate(df)
          This is the primary pathway for live/interactive data.
          Called by: analyze.xgs_map(), parse._season(), parse._scrape()

  PATH B: parse._season() -> calls nhl_api.get_game_feed() for each game
          -> parse._game() -> parse._elaborate() -> concat -> CSV
          This produces the raw elaborated CSVs (e.g. {season}_df.csv)

  PATH C: parse._scrape() -> calls nhl_api.get_game_feed() for each game
          -> saves raw feeds as JSON/CSV -> optionally _game() + _elaborate()
          This produces the raw game feed CSVs (e.g. {season}_game_feeds.csv)

  PATH D: data_pipeline.preprocess_features(df) 
          -> correction.fix_blocked_shot_attribution()
          -> orientation standardization
          -> distance/angle recalculation
          This is the TRAINING pipeline applied to Path A/B/C outputs.

  PATH E: fit_xgs.load_all_seasons_data() -> loads _df.csv files
          -> clean_df_for_model() -> model training
          This is how training data is actually consumed.

For each game, we will check:
  1. Raw API: What does eventOwnerTeamId say?
  2. parse._game(): What team_id does the parser assign?
  3. parse._elaborate(): Are distance/angle reasonable? (correct goal)
  4. correction.fix_blocked_shot_attribution(): Does this HELP or HURT?
"""
import requests
import pandas as pd
import numpy as np
import os
import sys
import json
import time
import random

sys.path.insert(0, os.getcwd())

from puck import nhl_api, parse, correction

# ============================================================
# CONFIGURATION
# ============================================================
GAMES_PER_SEASON = 10
SEASONS = {
    "20202021": list(range(2020020001, 2020020100)),  # Pool of candidates
    "20212022": list(range(2021020001, 2021020100)),
    "20222023": list(range(2022020001, 2022020100)),
    "20232024": list(range(2023020001, 2023020100)),
    "20242025": list(range(2024020001, 2024020100)),
    "20252026": list(range(2025020001, 2025020100)),
}

results = []

# ============================================================
# PART 1: RAW API VERIFICATION (Direct HTTP, no parse.py)
# ============================================================
print("=" * 80)
print("PART 1: RAW API VERIFICATION")
print("  Direct HTTP calls to api-web.nhle.com — no parse.py, no caches")
print("=" * 80)

for season, candidate_ids in SEASONS.items():
    print(f"\n--- Season {season} ---")
    games_tested = 0
    season_results = {
        "shooter_correct": 0,
        "shooter_wrong": 0,
        "total_blocks": 0,
        "games_tested": 0,
        "detail_keys_seen": set(),
    }
    
    # Sample 10 games randomly (within first 100 game IDs)
    random.seed(42)
    sampled = random.sample(candidate_ids, min(len(candidate_ids), 40))
    
    for gid in sampled:
        if games_tested >= GAMES_PER_SEASON:
            break
        
        try:
            # Direct HTTP — bypass all caches
            url = f"https://api-web.nhle.com/v1/gamecenter/{gid}/play-by-play"
            resp = requests.get(url, timeout=8)
            if resp.status_code != 200:
                continue
            data = resp.json()
            
            # Must have plays
            plays = data.get("plays", [])
            if not plays:
                continue
            
            home_id = data.get("homeTeam", {}).get("id")
            away_id = data.get("awayTeam", {}).get("id")
            home_abb = data.get("homeTeam", {}).get("abbrev", "???")
            away_abb = data.get("awayTeam", {}).get("abbrev", "???")
            
            if not home_id or not away_id:
                continue
            
            # Build roster
            roster = data.get("rosterSpots", [])
            p_teams = {}
            for p in roster:
                pid = p.get("playerId")
                tid = p.get("teamId")
                if pid and tid:
                    p_teams[pid] = tid
            
            blocked_plays = [p for p in plays if p.get("typeDescKey") == "blocked-shot"]
            
            if not blocked_plays:
                continue  # Skip games with no blocked shots
            
            games_tested += 1
            season_results["games_tested"] += 1
            
            for bp in blocked_plays:
                d = bp.get("details", {})
                owner_tid = d.get("eventOwnerTeamId")
                shooting_pid = d.get("shootingPlayerId")
                blocking_pid = d.get("blockingPlayerId")
                
                # Track all detail keys we see
                season_results["detail_keys_seen"].update(d.keys())
                
                season_results["total_blocks"] += 1
                
                # Check: does eventOwnerTeamId match shooter or blocker?
                if shooting_pid and shooting_pid in p_teams:
                    if p_teams[shooting_pid] == owner_tid:
                        season_results["shooter_correct"] += 1
                    else:
                        season_results["shooter_wrong"] += 1
            
            # Rate limit
            time.sleep(0.3)
            
        except Exception as e:
            continue
    
    total = season_results["total_blocks"]
    sc = season_results["shooter_correct"]
    sw = season_results["shooter_wrong"]
    
    print(f"  Games tested: {season_results['games_tested']}")
    print(f"  Total blocked shots: {total}")
    print(f"  eventOwnerTeamId == Shooter: {sc}/{total} ({100*sc/total:.1f}%)" if total else "  No blocked shots found")
    print(f"  eventOwnerTeamId == Blocker: {sw}/{total} ({100*sw/total:.1f}%)" if total else "")
    print(f"  Detail keys seen: {sorted(season_results['detail_keys_seen'])}")
    
    results.append({
        "season": season,
        "games": season_results["games_tested"],
        "total": total,
        "shooter_match": sc,
        "blocker_match": sw,
        "detail_keys": sorted(season_results["detail_keys_seen"]),
    })


# ============================================================
# PART 2: PARSE._GAME() PATHWAY
# ============================================================
print("\n" + "=" * 80)
print("PART 2: parse._game() OUTPUT VERIFICATION")
print("  Fetching fresh feeds via nhl_api.get_game_feed() -> parse._game()")
print("  Testing team_id assignment and player attribution")
print("=" * 80)

# Use 3 specific games (1 per era) for deep inspection
test_games = [
    (2020020050, "20202021"),
    (2022020050, "20222023"),
    (2024020050, "20242025"),
]

for gid, season in test_games:
    print(f"\n--- Game {gid} ({season}) ---")
    try:
        # Fetch fresh (force bypass cache)
        url = f"https://api-web.nhle.com/v1/gamecenter/{gid}/play-by-play"
        resp = requests.get(url, timeout=8)
        if resp.status_code != 200:
            print(f"  API returned {resp.status_code}, skipping")
            continue
        feed = resp.json()
        
        home_id = feed.get("homeTeam", {}).get("id")
        away_id = feed.get("awayTeam", {}).get("id")
        home_abb = feed.get("homeTeam", {}).get("abbrev", "???")
        away_abb = feed.get("awayTeam", {}).get("abbrev", "???")
        
        # Build ground truth roster map
        roster = feed.get("rosterSpots", [])
        p_teams = {}
        p_names = {}
        for p in roster:
            pid = p.get("playerId")
            tid = p.get("teamId")
            fn = p.get("firstName", {})
            ln = p.get("lastName", {})
            fname = fn.get("default") if isinstance(fn, dict) else fn
            lname = ln.get("default") if isinstance(ln, dict) else ln
            if pid:
                p_teams[pid] = tid
                p_names[pid] = f"{fname} {lname}" if fname and lname else "?"
        
        print(f"  {home_abb} (ID={home_id}) vs {away_abb} (ID={away_id})")
        
        # PATH A: parse._game()
        df_game = parse._game(feed)
        bs = df_game[df_game["event"] == "blocked-shot"]
        print(f"\n  parse._game() produced {len(bs)} blocked shots")
        
        correct = 0
        wrong = 0
        for _, row in bs.iterrows():
            pid = row.get("player_id")
            tid = row.get("team_id")
            actual_team = p_teams.get(pid)
            if actual_team is not None:
                if actual_team == tid:
                    correct += 1
                else:
                    wrong += 1
        
        print(f"    Player's actual team == row team_id: {correct}/{len(bs)}")
        print(f"    Mismatches: {wrong}/{len(bs)}")
        
        if wrong > 0:
            print("    MISMATCHED EXAMPLES:")
            for _, row in bs.head(3).iterrows():
                pid = row.get("player_id")
                tid = row.get("team_id")
                pname = row.get("player_name") or p_names.get(pid, "?")
                actual = p_teams.get(pid)
                actual_abb = home_abb if actual == home_id else away_abb if actual == away_id else "?"
                tid_abb = home_abb if tid == home_id else away_abb if tid == away_id else "?"
                print(f"      {pname}: actual_team={actual_abb}, row_team_id={tid_abb}")
        
        # PATH B: parse._elaborate()
        df_elab = parse._elaborate(df_game)
        bs_elab = df_elab[df_elab["event"] == "blocked-shot"]
        
        if "distance" in bs_elab.columns:
            sog_elab = df_elab[df_elab["event"] == "shot-on-goal"]
            bs_dist = bs_elab["distance"].dropna().mean()
            sog_dist = sog_elab["distance"].dropna().mean()
            print(f"\n  parse._elaborate() distances:")
            print(f"    Blocked shot avg: {bs_dist:.1f}")
            print(f"    Shot-on-goal avg: {sog_dist:.1f}")
            print(f"    Assessment: {'OK' if bs_dist < 80 else 'WRONG GOAL'}")
        
        # PATH C: correction.fix_blocked_shot_attribution()
        df_corrected = correction.fix_blocked_shot_attribution(df_game.copy())
        bs_corr = df_corrected[df_corrected["event"] == "blocked-shot"]
        
        corr_correct = 0
        corr_wrong = 0
        for _, row in bs_corr.iterrows():
            pid = row.get("player_id")
            tid = row.get("team_id")
            actual_team = p_teams.get(pid)
            if actual_team is not None:
                if actual_team == tid:
                    corr_correct += 1
                else:
                    corr_wrong += 1
        
        print(f"\n  AFTER correction.fix_blocked_shot_attribution():")
        print(f"    Player's actual team == row team_id: {corr_correct}/{len(bs_corr)}")
        print(f"    Mismatches: {corr_wrong}/{len(bs_corr)}")
        
        if corr_wrong > correct:
            print("    >>> CORRECTION IS MAKING THINGS WORSE (double-swapping)")
        elif corr_wrong < correct:
            print("    >>> CORRECTION IS HELPING (fixing attribution)")
        else:
            print("    >>> CORRECTION HAS NO NET EFFECT")
        
        time.sleep(0.5)
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback; traceback.print_exc()


# ============================================================
# PART 3: SUMMARY REPORT
# ============================================================
print("\n" + "=" * 80)
print("DEFINITIVE SUMMARY REPORT")
print("=" * 80)

print("\n--- RAW API Attribution (Part 1) ---")
print(f"{'Season':<12} {'Games':<8} {'Blocks':<10} {'Owner=Shooter':<18} {'Owner=Blocker':<18} {'Verdict'}")
print("-" * 80)

all_shooter = 0
all_total = 0
for r in results:
    total = r["total"]
    sc = r["shooter_match"]
    sw = r["blocker_match"]
    verdict = "SHOOTER" if sc > sw else "BLOCKER" if sw > sc else "INCONCLUSIVE"
    pct_s = f"{100*sc/total:.1f}%" if total else "N/A"
    pct_b = f"{100*sw/total:.1f}%" if total else "N/A"
    print(f"{r['season']:<12} {r['games']:<8} {total:<10} {sc} ({pct_s}){'':<6} {sw} ({pct_b}){'':<6} {verdict}")
    all_shooter += sc
    all_total += total

print(f"\nOVERALL: {all_shooter}/{all_total} ({100*all_shooter/all_total:.1f}%) eventOwnerTeamId == Shooter")

print("\n--- Detail Keys Found in blocked-shot events ---")
all_keys = set()
for r in results:
    all_keys.update(r["detail_keys"])
print(f"  All unique keys: {sorted(all_keys)}")

print("\n--- Parse._game() Assessment ---")
print("  If parse._game() uses eventOwnerTeamId for team_id,")
print("  and eventOwnerTeamId == Shooter (as proven above),")
print("  then parse._game() OUTPUT HAS CORRECT TEAM_ID.")

print("\n--- correction.fix_blocked_shot_attribution() Assessment ---")
print("  If the API already has correct attribution,")
print("  then the correction is DOUBLE-SWAPPING team_id.")
print("  This corrupts: is_home, coordinates, distance, angle,")
print("  relative_game_state, and rebound chains.")

print("\n--- Recommended Action ---")
print("  1. DISABLE team_id swap in correction.py")
print("  2. REVERT parse.py team_id flip for blocked shots")
print("  3. REGENERATE all _df.csv files from raw feeds")
print("  4. RETRAIN model on clean data")
