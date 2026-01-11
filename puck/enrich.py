
import json
import logging
import requests
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
import time

# Configure logging
logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache/player_info")
API_URL_TEMPLATE = "https://api-web.nhle.com/v1/player/{}/landing"

class PlayerEnricher:
    """
    Enriches player data (e.g., shoots/catches, position) by fetching from NHL API
    and caching results locally.
    """
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        # Basic browser-like headers often help generic scraping even if API is public
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })

    def get_player_info(self, player_id: int) -> Dict[str, Any]:
        """
        Retrieves player info from cache or API.
        Returns a dict with keys like 'shootsCatches', 'position', etc.
        Returns empty dict on failure.
        """
        if not player_id:
            return {}

        cache_file = self.cache_dir / f"{player_id}.json"
        
        # 1. Check Cache
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read cache for player {player_id}: {e}")

        # 2. Fetch from API
        url = API_URL_TEMPLATE.format(player_id)
        try:
            # Minimal polite delay if doing bulk non-cached
            time.sleep(0.05) 
            resp = self.session.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                
                # Extract relevant fields to keep cache small(ish)
                extracted = {
                    'ok': True,
                    'last_updated': time.time(),
                    'shootsCatches': data.get('shootsCatches'),
                    'position': data.get('position'),
                    'heightInInches': data.get('heightInInches'),
                    'weightInPounds': data.get('weightInPounds'),
                    'birthDate': data.get('birthDate')
                }
                
                # Save to cache
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(extracted, f)
                
                return extracted
            else:
                logger.warning(f"API fetch failed for {player_id}: Status {resp.status_code}")
                # Cache failure? optional. For now, don't cache failure so we retry next time.
                return {'ok': False}
                
        except Exception as e:
            logger.error(f"Error fetching player {player_id}: {e}")
            return {'ok': False}

    def enrich_dataframe(self, df: pd.DataFrame, target_cols: List[str] = ['shoots_catches', 'shooter_role']) -> pd.DataFrame:
        """
        Iterates over DataFrame, finds rows with missing target columns (if they exist or are needed),
        fetches player data, and fills gaps.
        """
        if df.empty or 'player_id' not in df.columns:
            return df

        # Ensure target columns exist
        for col in target_cols:
            if col not in df.columns:
                df[col] = None

        # Identify missing data
        # We look for rows where player_id is present but target col is null or 'Unknown'
        # Currently mainly focused on 'shoots_catches'
        
        # Get unique player IDs that need lookup
        # Filter: player_id is not null AND (shoots_catches is missing OR 'Unknown')
        mask_missing = pd.Series(False, index=df.index)
        if 'shoots_catches' in target_cols:
            mask_missing = mask_missing | (df['shoots_catches'].isna()) | (df['shoots_catches'] == 'Unknown')
        if 'shooter_role' in target_cols:
            mask_missing = mask_missing | (df['shooter_role'].isna()) | (df['shooter_role'] == 'Unknown')
        
        # Only valid player IDs
        mask_valid_pid = df['player_id'].notna() & (df['player_id'] != 0)
        
        players_to_fetch = df.loc[mask_missing & mask_valid_pid, 'player_id'].unique()
        
        if len(players_to_fetch) == 0:
            return df

        logger.info(f"Enriching data for {len(players_to_fetch)} players...")

        # Bulk fetch/cache
        info_map = {}
        for pid in players_to_fetch:
            # Cast to int likely
            try:
                pid_int = int(pid)
                data = self.get_player_info(pid_int)
                if data.get('ok'):
                    info_map[pid] = data
            except Exception:
                continue

        # Apply updates
        # We could use map(), but let's iterate to be safe/explicit with column mapping
        # Or faster: create a mapping dict for shoots_catches and map it
        
        # Prepare Maps
        sc_map = {}
        role_map = {}
        
        for pid, d in info_map.items():
            # Shoots
            if d.get('shootsCatches'):
                sc_map[pid] = d.get('shootsCatches')
            
            # Role (From Position)
            pos = d.get('position')  # e.g. 'L', 'C', 'D', 'G'
            if pos:
                # Map: D -> D, G -> G, L/R/C -> F, else None/Skip
                role = 'D' if pos == 'D' else ('G' if pos == 'G' else ('F' if pos in ['L', 'R', 'C'] else None))
                if role:
                    role_map[pid] = role

        if 'shoots_catches' in target_cols and sc_map:
            mapped_values = df['player_id'].map(sc_map)
            # 1. Fill NaNs
            df['shoots_catches'] = df['shoots_catches'].fillna(mapped_values)
            # 2. Overwrite Unknowns
            mask_unknown = (df['shoots_catches'] == 'Unknown')
            if mask_unknown.any():
                 df.loc[mask_unknown, 'shoots_catches'] = mapped_values[mask_unknown]

        if 'shooter_role' in target_cols and role_map:
            mapped_roles = df['player_id'].map(role_map)
            # 1. Fill NaNs
            df['shooter_role'] = df['shooter_role'].fillna(mapped_roles)
            # 2. Overwrite Unknowns
            mask_unknown_role = (df['shooter_role'] == 'Unknown')
            if mask_unknown_role.any():
                 df.loc[mask_unknown_role, 'shooter_role'] = mapped_roles[mask_unknown_role]

        return df
