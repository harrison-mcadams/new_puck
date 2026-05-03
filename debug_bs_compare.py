import pandas as pd

# Check if the 2022-23 season was processed differently
for season in ["20222023", "20232024"]:
    df = pd.read_csv(f"data/{season}/{season}_df.csv", low_memory=False, nrows=500)
    bs = df[df["event"] == "blocked-shot"]
    if len(bs) > 0:
        row = bs.iloc[0]
        print(f"\n{season} sample blocked shot:")
        for k in ["event", "team_id", "home_id", "away_id", "x", "y",
                   "distance", "angle_deg", "home_team_defending_side",
                   "player_id", "player_name"]:
            v = row.get(k, "N/A")
            print(f"  {k}: {v}")
        
        # Check if distances were recalculated correctly
        sog = df[df["event"] == "shot-on-goal"]
        print(f"\n  Stats - BS count: {len(bs)}, SOG count: {len(sog)}")
        print(f"  BS  distance range: [{bs['distance'].min():.1f}, {bs['distance'].max():.1f}]")
        print(f"  SOG distance range: [{sog['distance'].min():.1f}, {sog['distance'].max():.1f}]")
