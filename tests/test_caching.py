import unittest
from unittest.mock import patch, MagicMock
import shutil
import tempfile
from pathlib import Path
import pandas as pd
import time
import os
import sys

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import timing
import nhl_api
import parse

class TestShiftCaching(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for data/season/shifts
        self.test_dir = tempfile.mkdtemp()
        self.game_id = 2025020001
        self.season = "20252026"
        
        # We need to patch the path construction in timing.py or control where it writes.
        # Since I haven't written the code yet, I will design the test to expect 
        # the cache to be at data/20252026/shifts.
        # To make this testable without messing up real data, I might need to 
        # allow configuring the data root or mock Path.
        # For now, I'll mock the specific cache path logic or just use a mock side_effect 
        # if I can't easily inject the path.
        
        # Actually, the plan says `data/{season}/shifts`. 
        # I will mock `pathlib.Path` or just let it write to a temp dir by 
        # patching the base data directory if possible, but `timing.py` likely uses relative paths.
        # A better approach for the test is to patch `timing.Path` or the specific variable if I make it a constant.
        pass

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch('timing.nhl_api.get_shifts')
    @patch('timing.parse._shifts')
    def test_caching_behavior(self, mock_parse_shifts, mock_get_shifts):
        # Setup mocks
        mock_get_shifts.return_value = {'some': 'data'}
        
        # Create a dummy dataframe to return
        dummy_df = pd.DataFrame({
            'game_id': [self.game_id],
            'start_total_seconds': [100.0],
            'end_total_seconds': [145.0],
            'player_id': [8478402],
            'team_id': [16],
            'period': [1]
        })
        mock_parse_shifts.return_value = dummy_df

        # We need to intercept the cache path creation to point to our temp dir
        # or we can just mock the file operations?
        # Let's try to patch `timing.Path` is risky.
        # Instead, I'll implement `timing.py` to respect a module-level `DATA_DIR` 
        # or similar that I can patch.
        # Or I can just patch `pd.read_pickle` and `pd.DataFrame.to_pickle` 
        # and verify they are called with expected paths.
        
        with patch('timing.pd.read_pickle') as mock_read_pickle, \
             patch('timing.pd.DataFrame.to_pickle') as mock_to_pickle, \
             patch('timing.Path.exists') as mock_exists:
            
            # 1. First call: Cache does not exist
            mock_exists.return_value = False
            
            # Simulate slow API call
            def side_effect(*args, **kwargs):
                time.sleep(0.1)
                return {'some': 'data'}
            mock_get_shifts.side_effect = side_effect

            start_time = time.time()
            df1 = timing._get_shifts_df(self.game_id)
            duration1 = time.time() - start_time
            
            # Verify API was called and data was saved
            mock_get_shifts.assert_called_once()
            mock_to_pickle.assert_called_once()
            self.assertTrue(duration1 >= 0.1)
            
            # Verify the path passed to to_pickle looks correct (ends with shifts_2025020001.pkl)
            args, _ = mock_to_pickle.call_args
            self.assertTrue(str(args[0]).endswith(f'shifts_{self.game_id}.pkl'))
            
            # 2. Second call: Cache exists
            mock_exists.return_value = True
            mock_read_pickle.return_value = dummy_df
            mock_get_shifts.reset_mock()
            
            start_time = time.time()
            df2 = timing._get_shifts_df(self.game_id)
            duration2 = time.time() - start_time
            
            # Verify API was NOT called and data was loaded
            mock_get_shifts.assert_not_called()
            mock_read_pickle.assert_called_once()
            self.assertTrue(duration2 < 0.1)
            
            pd.testing.assert_frame_equal(df1, df2)

if __name__ == '__main__':
    unittest.main()
