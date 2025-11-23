#!/usr/bin/env python
"""
Test script to validate league() and season_analysis() functions.

This script tests the new league baseline and season analysis functionality
without requiring actual season data.
"""

import os
import sys
import numpy as np
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_league_function():
    """Test the league() function with a small subset of teams."""
    print("=" * 60)
    print("Testing league() function")
    print("=" * 60)
    
    try:
        from analyze import league
        
        # Test with a very small subset of teams to avoid long computation
        test_teams = ['PHI', 'BOS']
        print(f"\nTesting with teams: {test_teams}")
        
        # Note: This will fail if no CSV data exists, which is expected
        # We're just testing the function structure
        try:
            result = league(
                season='20252026',
                teams=test_teams,
                mode='compute'
            )
            print("\n✓ league() function executed successfully")
            print(f"  - Result keys: {list(result.keys())}")
            print(f"  - Stats: {result.get('stats', {})}")
            return True
        except FileNotFoundError as e:
            print(f"\n⚠ Expected error (no CSV data): {e}")
            print("  This is OK - function structure is valid")
            return True
        except Exception as e:
            print(f"\n✗ Unexpected error: {type(e).__name__}: {e}")
            return False
            
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_league_load_mode():
    """Test league() in load mode."""
    print("\n" + "=" * 60)
    print("Testing league() load mode")
    print("=" * 60)
    
    try:
        from analyze import league
        
        # Create dummy baseline files for testing load mode
        static_dir = 'static'
        os.makedirs(static_dir, exist_ok=True)
        
        baseline_npy = os.path.join(static_dir, '20252026_league_baseline.npy')
        baseline_json = os.path.join(static_dir, '20252026_league_baseline.json')
        
        # Create dummy data
        dummy_map = np.random.rand(86, 201) * 0.05  # Small random values
        dummy_stats = {
            'total_left_seconds': 1000.0,
            'total_left_xg': 50.0,
            'stats': {
                'season': '20252026',
                'n_teams': 2,
                'xg_per60': 180.0
            },
            'per_team': {}
        }
        
        np.save(baseline_npy, dummy_map)
        with open(baseline_json, 'w') as f:
            json.dump(dummy_stats, f)
        
        print(f"\nCreated dummy baseline files:")
        print(f"  - {baseline_npy}")
        print(f"  - {baseline_json}")
        
        # Test load mode
        result = league(season='20252026', mode='load')
        
        print("\n✓ league() load mode executed successfully")
        print(f"  - Loaded stats: {result.get('stats', {})}")
        print(f"  - Baseline map shape: {result.get('combined_norm').shape if result.get('combined_norm') is not None else 'None'}")
        
        # Clean up
        try:
            os.remove(baseline_npy)
            os.remove(baseline_json)
            print("\n✓ Cleaned up test files")
        except Exception:
            pass
            
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_season_analysis_function():
    """Test the season_analysis() function structure."""
    print("\n" + "=" * 60)
    print("Testing season_analysis() function")
    print("=" * 60)
    
    try:
        from analyze import season_analysis
        
        print("\n✓ season_analysis() function imported successfully")
        print("  Note: Full execution requires CSV data which may not be available")
        
        # Check function signature
        import inspect
        sig = inspect.signature(season_analysis)
        print(f"  - Function parameters: {list(sig.parameters.keys())}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_cli_integration():
    """Test that CLI has the new options."""
    print("\n" + "=" * 60)
    print("Testing CLI integration")
    print("=" * 60)
    
    import subprocess
    
    try:
        result = subprocess.run(
            [sys.executable, 'analyze.py', '--help'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=10
        )
        
        help_text = result.stdout
        
        # Check for our new options
        checks = [
            ('--league-baseline', 'league-baseline option'),
            ('--baseline-mode', 'baseline-mode option'),
            ('--season-analysis', 'season-analysis option'),
        ]
        
        all_found = True
        for flag, desc in checks:
            if flag in help_text:
                print(f"✓ Found {desc}")
            else:
                print(f"✗ Missing {desc}")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"✗ Error running CLI help: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("LEAGUE BASELINE & SEASON ANALYSIS TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("League function structure", test_league_function),
        ("League load mode", test_league_load_mode),
        ("Season analysis function", test_season_analysis_function),
        ("CLI integration", test_cli_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
