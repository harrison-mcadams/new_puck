
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

from puck import rink
from puck.plot import plot_relative_map
from puck.spline_transformer import TensorSpline

def generate_spatial_grids(model, output_dir, teams: list = None, df: pd.DataFrame = None):
    """
    Generates spatial impact maps (Marginal Log-Odds) for all teams.
    If df is provided, calculates summary stats (Goals, xG).
    """
    print("WARNING: mixed_effects_viz.generate_spatial_grids is currently disabled due to file corruption.")
    return

    # Original code was corrupted.
    # ...


