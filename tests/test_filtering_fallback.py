
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import sys
import os
import numpy as np

# Add parent directory to path to import analyze
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analyze

class TestFilteringFallback(unittest.TestCase):
    def setUp(self):
        # Create a dummy dataframe
        self.df = pd.DataFrame({
            'game_id': [2025020001] * 10,
            'total_time_elapsed_seconds': np.arange(10) * 60.0, # 0, 60, 120...
            'game_state': ['5v5'] * 5 + ['5v4'] * 5,
            'event': ['shot-on-goal'] * 10,
            'x': [0] * 10,
            'y': [0] * 10,
            'team_id': [1] * 10,
            'home_id': [1] * 10,
            'away_id': [2] * 10
        })
        
        # Mock timing module within analyze
        self.timing_patcher = patch('analyze.timing')
        self.mock_timing = self.timing_patcher.start()

    def tearDown(self):
        self.timing_patcher.stop()

    def test_fallback_when_intervals_missing(self):
        """
        Test that xgs_map falls back to condition filtering when interval data is missing/empty.
        """
        # Setup mock to return empty intervals (simulating missing data)
        self.mock_timing.compute_game_timing.return_value = {'per_game': {}}
        
        condition = {'game_state': ['5v5']}
        
        with patch('analyze._predict_xgs', side_effect=lambda df, **kwargs: (df, None, None)):
            res = analyze.xgs_map(
                data_df=self.df,
                condition=condition,
                return_filtered_df=True,
                show=False,
                out_path='test_fallback.png'
            )
            
            # Unpack result
            df_filtered = res[2]
            
            # Should have filtered to 5v5 using condition (5 rows)
            # If fallback didn't happen, it would use empty intervals and return 0 rows
            self.assertEqual(len(df_filtered), 5, "Should fallback to condition filtering and return 5 rows")

    def test_no_fallback_when_intervals_present_but_no_match(self):
        """
        Test that xgs_map does NOT fallback when interval data is present but yields 0 matches.
        """
        # Setup mock to return valid intervals that don't match the data
        self.mock_timing.compute_game_timing.return_value = {
            'per_game': {
                2025020001: {
                    'intervals_per_condition': {'5v5': [(1000.0, 2000.0)]},
                    'sides': {'team': {'intersection_intervals': [(1000.0, 2000.0)]}}
                }
            }
        }
        
        condition = {'game_state': ['5v5']}
        
        with patch('analyze._predict_xgs', side_effect=lambda df, **kwargs: (df, None, None)):
            res = analyze.xgs_map(
                data_df=self.df,
                condition=condition,
                return_filtered_df=True,
                show=False,
                out_path='test_fallback.png'
            )
            
            # Unpack result
            df_filtered = res[2]
            
            # Should be empty because intervals don't match, and we should NOT fallback
            self.assertTrue(df_filtered.empty, "Should return empty df when intervals don't match")

if __name__ == '__main__':
    unittest.main()
