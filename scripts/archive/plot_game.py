#!/usr/bin/env python3
"""CLI script to plot shot map for a single game using the puck package."""

import argparse
import sys
import os

# Add project root to sys.path to allow importing puck package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import analyze

def main():
    parser = argparse.ArgumentParser(description='Plot shot map for a single game')
    # Use 'game_id' as positional argument, but allow flexibility
    parser.add_argument('game_id', help='NHL Game ID (e.g. 2024020123)')
    parser.add_argument('--season', default='20252026', help='Season string (default: 20252026)')
    parser.add_argument('--out', '-o', help='Output file path (default: shot_plot.png)', default='shot_plot.png')
    
    args = parser.parse_args()

    print(f"Plotting game {args.game_id} (Season {args.season})...")
    
    try:
        # Call xgs_map from analyze module
        # xgs_map returns tuple: (out_path, heatmap, filtered_df, stats)
        # We just want the plot.
        out_path, _, _, stats = analyze.xgs_map(
            season=args.season, 
            game_id=args.game_id, 
            out_path=args.out, 
            show=False
        )
        
        if out_path:
            print(f"Success! Saved plot to {out_path}")
            if stats:
                print(f"Stats: Home xG: {stats.get('team_xgs', 0):.2f}, Away xG: {stats.get('other_xgs', 0):.2f}")
        else:
            print("Warning: xgs_map returned no output path. The plot might not have been generated.")
            
    except Exception as e:
        print(f"Error generating plot: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
