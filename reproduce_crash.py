
import sys
import os
import logging

# Ensure we can import puck
sys.path.append(os.getcwd())

from puck import analyze
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reproduce():
    game_id = 2024010074 # BOS vs PHI
    out_path = "analysis/reproduce_plot.png"
    
    logger.info(f"Attempting to plot game {game_id}...")
    
    try:
        ret = analyze.xgs_map(
            game_id=game_id,
            condition={},
            out_path=out_path,
            show=False,
            return_heatmaps=False,
            events_to_plot=['shot-on-goal', 'goal', 'xgs'],
            return_filtered_df=True,
            force_refresh=True
        )
        logger.info("Plot generated successfully!")
        
    except Exception as e:
        logger.exception("Crash during plotting due to: %s", e)

if __name__ == "__main__":
    reproduce()
