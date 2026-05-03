"""
Diagnostic script: Investigate blocked-shot attribution across the NHL API
and the existing training CSVs.
"""
import requests
import json
import pandas as pd
import os
import glob

# ============================================================
# PART 1: Raw API Investigation
# ============================================================
print("=" * 70)
print("PART 1: RAW API STRUCTURE FOR BLOCKED SHOTS")
print("=" * 70)

game_id = 2025021224
r = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play")
data = r.json()

home_id = data["homeTeam"]["id"]
away_id = data["awayTeam"]["id"]
home_abb = data["homeTeam"]["abbrev"]
away_abb = data["awayTeam"]["abbrev"]
print(f"Home: {home_abb} (ID={home_id}), Away: {away_abb} (ID={away_id})")
print()

# Build roster map: player_id -> (name, team_id)
roster = data.get("rosterSpots", [])
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

# Analyze blocked shots
plays = [p for p in data.get("plays", []) if p.get("typeDescKey") == "blocked-shot"]
print(f"Found {len(plays)} blocked-shot events\n")

for i, p in enumerate(plays[:5]):
    d = p.get("details", {})
    owner_tid = d.get("eventOwnerTeamId")
    owner_team = home_abb if owner_tid == home_id else away_abb
    
    print(f"--- Blocked Shot #{i+1} ---")
    print(f"  eventOwnerTeamId: {owner_tid} ({owner_team})")
    
    # Check all player-related keys
    for key in d:
        if "layer" in key.lower():
            pid = d[key]
            info = player_info.get(pid, {"name": "UNKNOWN", "team": "?"})
            print(f"  {key}: {pid} -> {info['name']} ({info['team']})")
    
    print(f"  xCoord: {d.get('xCoord')}, yCoord: {d.get('yCoord')}")
    print()

# Summarize the pattern
print("\n--- SUMMARY ---")
shooter_matches_owner = 0
blocker_matches_owner = 0
for p in plays:
    d = p.get("details", {})
    owner_tid = d.get("eventOwnerTeamId")
    
    shooting_pid = d.get("shootingPlayerId")
    blocking_pid = d.get("blockingPlayerId")
    
    if shooting_pid and shooting_pid in player_info:
        if player_info[shooting_pid]["team_id"] == owner_tid:
            shooter_matches_owner += 1
    
    if blocking_pid and blocking_pid in player_info:
        if player_info[blocking_pid]["team_id"] == owner_tid:
            blocker_matches_owner += 1

print(f"  Shooter's team == eventOwnerTeamId: {shooter_matches_owner}/{len(plays)}")
print(f"  Blocker's team == eventOwnerTeamId: {blocker_matches_owner}/{len(plays)}")
if blocker_matches_owner > shooter_matches_owner:
    print("  >>> eventOwnerTeamId is the BLOCKER'S team (defensive team)")
    print("  >>> We MUST flip team_id to shooter's team for shot attribution!")
elif shooter_matches_owner > blocker_matches_owner:
    print("  >>> eventOwnerTeamId is the SHOOTER'S team (offensive team)")
    print("  >>> Current attribution is CORRECT for shot attempts.")
else:
    print("  >>> INCONCLUSIVE - need more data")


# ============================================================
# PART 2: Check existing training CSVs
# ============================================================
print("\n" + "=" * 70)
print("PART 2: CHECKING EXISTING TRAINING CSVs")
print("=" * 70)

data_dir = "data"
seasons = ["20202021", "20212022", "20222023", "20232024", "20242025", "20252026"]

for season in seasons:
    csv_path = os.path.join(data_dir, season, f"season_{season}.csv")
    if not os.path.exists(csv_path):
        # Try alternate patterns
        csvs = glob.glob(os.path.join(data_dir, season, "*.csv"))
        if csvs:
            csv_path = csvs[0]
        else:
            print(f"\n  {season}: No CSV found, skipping")
            continue
    
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        print(f"\n  {season}: Error reading CSV: {e}")
        continue
    
    blocked = df[df["event"] == "blocked-shot"] if "event" in df.columns else pd.DataFrame()
    total = len(blocked)
    
    if total == 0:
        print(f"\n  {season}: No blocked-shot events found in CSV")
        continue
    
    print(f"\n  {season}: {total} blocked-shot events")
    
    # Check: does team_id match home_id more often, or away_id?
    if "team_id" in df.columns and "home_id" in df.columns:
        bs_home = (blocked["team_id"].astype(str) == blocked["home_id"].astype(str)).sum()
        bs_away = (blocked["team_id"].astype(str) == blocked["away_id"].astype(str)).sum()
        print(f"    team_id == home_id: {bs_home} ({100*bs_home/total:.1f}%)")
        print(f"    team_id == away_id: {bs_away} ({100*bs_away/total:.1f}%)")
        
        # Compare with shots-on-goal distribution (should be ~50/50)
        sog = df[df["event"] == "shot-on-goal"]
        if len(sog) > 0:
            sog_home = (sog["team_id"].astype(str) == sog["home_id"].astype(str)).sum()
            sog_pct = 100 * sog_home / len(sog)
            print(f"    (For reference: shot-on-goal home%: {sog_pct:.1f}%)")
    
    # Check if x_adj or distance columns exist and whether blocked shots
    # have similar spatial distribution to regular shots
    if "distance" in df.columns:
        sog_dist = df[df["event"] == "shot-on-goal"]["distance"].dropna().mean()
        bs_dist = blocked["distance"].dropna().mean()
        goal_dist = df[df["event"] == "goal"]["distance"].dropna().mean() if "goal" in df["event"].values else float("nan")
        print(f"    Avg distance - SOG: {sog_dist:.1f}, Blocked: {bs_dist:.1f}, Goal: {goal_dist:.1f}")

print("\n" + "=" * 70)
print("PART 3: CONCLUSION")
print("=" * 70)
print("If eventOwnerTeamId is the BLOCKER'S team, then all CSVs have")
print("blocked shots attributed to the WRONG team. This would mean:")
print("  - Distance/angle features are computed from wrong goal")
print("  - is_home flag is flipped for ~20% of shot attempts")
print("  - Game state (PP/PK) perspective is reversed")
print("  - Rebound/rush chains may be broken")
print()
print("NEXT STEPS: If contamination confirmed, re-scrape all seasons.")
