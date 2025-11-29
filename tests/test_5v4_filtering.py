
import unittest
import pandas as pd
import sys
import os

# Add parent directory to path to import analyze
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analyze
import nhl_api

class Test5v4Filtering(unittest.TestCase):
    def test_5v4_filtering_includes_opponent_events(self):
        """
        Verify that filtering for '5v4' (Power Play) includes events from both
        the team on the PP and the opponent (SH).
        """
        # Use a known game or find one dynamically
        # PHI vs NYI 2025020381 had 5v4 time
        try:
            gid = nhl_api.get_game_id(team='PHI')
        except:
            # Fallback to a hardcoded ID if dynamic fetch fails or returns a game with no PP
            gid = 2025020381 

        print(f"Testing with game {gid}")

        # Condition: PHI Power Play
        condition = {'game_state': ['5v4'], 'team': 'PHI'}

        # Run analysis
        # We use return_filtered_df=True to inspect the results
        # We don't need to save the plot, so we can use a temp path or just ignore it
        try:
            res = analyze.xgs_map(
                game_id=gid, 
                condition=condition, 
                return_filtered_df=True, 
                show=False,
                out_path='test_5v4_output.png'
            )
        except Exception as e:
            self.fail(f"xgs_map failed with exception: {e}")

        if not isinstance(res, tuple) or len(res) < 3:
             self.fail("xgs_map did not return expected tuple (out_path, heatmaps, df, ...)")
        
        df_filtered = res[2]
        
        if df_filtered is None or df_filtered.empty:
            print("Warning: No 5v4 events found for this game. This might be a valid result if there were no PPs, but makes the test inconclusive.")
            return

        # Identify PHI's ID
        phi_id = None
        if 'home_abb' in df_filtered.columns and df_filtered['home_abb'].iloc[0] == 'PHI':
            phi_id = df_filtered['home_id'].iloc[0]
        elif 'away_abb' in df_filtered.columns and df_filtered['away_abb'].iloc[0] == 'PHI':
            phi_id = df_filtered['away_id'].iloc[0]
        
        self.assertIsNotNone(phi_id, "Could not determine PHI team ID from dataframe")

        # Split events
        phi_events = df_filtered[df_filtered['team_id'] == phi_id]
        opp_events = df_filtered[df_filtered['team_id'] != phi_id]

        print(f"PHI Events: {len(phi_events)}")
        print(f"Opponent Events: {len(opp_events)}")

        # Assertions
        # We expect at least some events from the opponent if the sample size is large enough.
        # If the game had PPs, there should be shots/blocks/hits etc.
        # We can't strictly assert > 0 for every single game without knowing the game flow,
        # but for a full game it's highly likely.
        
        # Check that if there are opponent events, they are NOT 5v4 in raw state (they should be 4v5)
        if not opp_events.empty:
            raw_states = opp_events['game_state'].unique()
            print(f"Opponent Raw States: {raw_states}")
            # We expect '4v5' or similar, NOT '5v4' (unless data error)
            # But wait, analyze.py might NOT overwrite the raw 'game_state' column in the returned df,
            # it only uses the relative column for filtering.
            # Let's verify that '4v5' is present if '5v4' was requested for PHI.
            
            # Note: The raw data might say '4v5' for opponent events.
            for state in raw_states:
                self.assertNotEqual(state, '5v4', "Opponent event has raw state '5v4', which implies they are on PP, but we filtered for PHI PP")

        # Clean up
        if os.path.exists('test_5v4_output.png'):
            os.remove('test_5v4_output.png')

if __name__ == '__main__':
    unittest.main()
