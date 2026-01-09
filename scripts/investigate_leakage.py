import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import fit_xgs, features, nhl_api, impute

def check_leakage():
    # Load 2025-2026 data
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    season = '20252026'
    path = data_dir / f'{season}.csv'
    
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    
    # Limit for speed (PBP scraping needed for enrichment)
    games = df['game_id'].unique()
    if len(games) > 50:
        print(f"Limiting to first 50 games for detailed analysis out of {len(games)}...")
        games = games[:50]
        df = df[df['game_id'].isin(games)].copy()
        
    print(f"Analyzing {len(df)} events...")

    # --- ENRICHMENT (Shooter ID + Role) ---
    print("Enriching Blocked Shots with Shooter Info (PBP fetch)...")
    season_str = str(season)
    season_full = f"{season_str}{int(season_str)+1}" if len(season_str)==4 else season_str
    
    try:
        bios = nhl_api.get_season_player_bios(season=season_full)
    except:
        bios = {}
    
    # Add 'shooter_id' column if missing
    if 'shooter_id' not in df.columns:
        df['shooter_id'] = np.nan
        
    # Standard shots have shooter_id = player_id
    mask_shot = df['event'].isin(['shot-on-goal', 'missed-shot', 'goal'])
    df.loc[mask_shot, 'shooter_id'] = df.loc[mask_shot, 'player_id']
    
    # We iterate GAMES to modify blocks efficiently
    processed_games = 0
    unique_games = df['game_id'].unique()
    
    for gid in unique_games:
        # Get blocks for this game
        g_blocks = df[(df['game_id'] == gid) & (df['event'] == 'blocked-shot')]
        if g_blocks.empty:
            continue
            
        try:
            pbp = nhl_api.get_game_feed(gid)
            if not pbp: continue
            
            plays = pbp.get('plays', [])
            mapping = {}
            for p in plays:
                if p.get('typeDescKey') == 'blocked-shot':
                    eid = p.get('eventId')
                    sid = p.get('details', {}).get('shootingPlayerId')
                    if eid and sid:
                        mapping[int(eid)] = sid
            
            # Apply to DF
            for idx, row in g_blocks.iterrows():
                eid = row.get('event_id')
                if pd.notna(eid):
                   sid = mapping.get(int(eid))
                   if sid:
                       df.at[idx, 'shooter_id'] = sid
                       
        except Exception as e:
            pass
        processed_games += 1
        if processed_games % 10 == 0:
            print(f"Processed {processed_games} games...")

    # Define Role Helper
    def get_role(pid):
        try:
            pid = int(pid) 
            if pid in bios:
                return 'D' if bios[pid].get('positionCode') == 'D' else 'F'
            if str(pid) in bios:
                return 'D' if bios[str(pid)].get('positionCode') == 'D' else 'F'
        except:
            pass
        return 'F' # Default to F
        
    print("Deriving roles...")
    df['shooter_role'] = df['shooter_id'].apply(get_role)
    
    # --- APPLY IMPUTATION ---
    print("Running Imputations (Split Model)...")
    # This will update 'distance' and 'angle_deg' for blocked shots
    # and populate 'imputed_x/y'
    df_imputed = impute.impute_blocked_shot_origins(df, role_col='shooter_role')
    
    # Use the IMPUTED dataframe for analysis
    df = df_imputed
    
    # Filter for standard play
    mask_block = df['event'] == 'blocked-shot'
    mask_shot = df['event'].isin(['shot-on-goal', 'missed-shot', 'goal'])
    
    df_blocks = df[mask_block].copy()
    df_shots = df[mask_shot].copy()
    
    print(f"Blocked Shots: {len(df_blocks)}")
    print(f"Unblocked Shots: {len(df_shots)}")
    
    # Helper for "role" used in plotting code
    df['role'] = df['shooter_role']
    df_blocks = df[mask_block].copy()
    df_shots = df[mask_shot].copy()
    
    print("\n--- Distance Statistics (Post-Imputation) ---")
    print("Blocked Shots Distance:")
    print(df_blocks['distance'].describe())
    print("\nUnblocked Shots Distance:")
    print(df_shots['distance'].describe())
    
    avg_block = df_blocks['distance'].mean()
    avg_shot = df_shots['distance'].mean()
    print(f"Mean Difference: {avg_shot - avg_block:.2f}")

    # Plotting
    try:
        # 1. Distance Histogram
        plt.figure(figsize=(10, 6))
        plt.hist(df_shots['distance'].dropna(), bins=50, alpha=0.5, label='Unblocked Shots', density=True, color='blue')
        plt.hist(df_blocks['distance'].dropna(), bins=50, alpha=0.5, label='Blocked Shots', density=True, color='red')
        plt.xlabel('Distance from Net (ft)')
        plt.ylabel('Density')
        plt.title('Leakage Check: Blocked Shot Distance Distribution (Post-Imputation)')
        plt.legend()
        out_dist = 'analysis/leakage_distance_distribution_fixed.png'
        plt.savefig(out_dist)
        print(f"Saved distance plot to {out_dist}")
        plt.close()
        
        # 2. Role vs Distance
        # Filter to known roles
        df_clean = df[df['role'].isin(['F', 'D'])].copy()
        mask_clean_shot = df_clean['event'].isin(['shot-on-goal', 'missed-shot', 'goal'])
        mask_clean_block = df_clean['event'] == 'blocked-shot'
        
        data_to_plot = [
            df_clean[(df_clean['role'] == 'D') & (mask_clean_shot)]['distance'].dropna(),
            df_clean[(df_clean['role'] == 'F') & (mask_clean_shot)]['distance'].dropna(),
            df_clean[(df_clean['role'] == 'D') & (mask_clean_block)]['distance'].dropna(),
            df_clean[(df_clean['role'] == 'F') & (mask_clean_block)]['distance'].dropna()
        ]
        labels = ['Unblocked (D)', 'Unblocked (F)', 'Blocked (D)', 'Blocked (F)']
        
        plt.figure(figsize=(10, 6))
        plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
        plt.title('Role vs Distance (Post-Imputation)')
        plt.ylabel('Distance (ft)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        out_role = 'analysis/leakage_role_distance_fixed.png'
        plt.savefig(out_role)
        print(f"Saved role plot to {out_role}")
        plt.close()
        
        # 3. "Impossible Event" Frequency Check
        d_in_slot_blocks = len(df_blocks[(df_blocks['role']=='D') & (df_blocks['distance'] < 30)])
        d_in_slot_shots = len(df_shots[(df_shots['role']=='D') & (df_shots['distance'] < 30)])
        
        total_blocks = len(df_blocks)
        total_shots = len(df_shots)
        
        print("\n--- The 'Impossible' Scenario Check (Defenseman < 30ft) ---")
        print(f"Blocks by D in Slot: {d_in_slot_blocks} ({d_in_slot_blocks/total_blocks:.1%} of all blocks)")
        print(f"Shots by D in Slot:  {d_in_slot_shots} ({d_in_slot_shots/total_shots:.1%} of all shots)")

    except Exception as e:
        print(f"Plotting failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_leakage()
