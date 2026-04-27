import os
import sys
import argparse
import logging

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import playoffs

def main():
    parser = argparse.ArgumentParser(description="Generate Playoff Plots")
    parser.add_argument('--season', type=str, default='20252026', help='Season string')
    parser.add_argument('--force', action='store_true', help='Force regeneration of plots')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print(f"--- Generating Playoff Plots for {args.season} ---")
    playoffs.generate_playoff_plots(season=args.season, force=args.force)
    print("--- Done ---")

if __name__ == "__main__":
    main()
