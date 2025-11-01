import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

RINK_LENGTH = 200
RINK_WIDTH = 85

def draw_rink(ax=None, center_line=True):
    """Draw a simplified NHL rink on the given matplotlib Axes.

    Coordinates use the NHL API convention: origin at center ice, x to the right (+100 to -100),
    y up/down from center (-42.5 to 42.5).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.5))
    # rink outline (centered at 0,0): x from -100..100, y from -42.5..42.5
    rink = patches.Rectangle((-100, -42.5), RINK_LENGTH, RINK_WIDTH,
                             linewidth=1, edgecolor="black", facecolor="#f0f8ff", zorder=0)
    ax.add_patch(rink)
    # center line
    if center_line:
        ax.plot([0, 0], [-42.5, 42.5], color="red", linewidth=1)
    # goal lines (approx at +/-89 feet)
    ax.plot([-89, -89], [-42.5, 42.5], color="red", linewidth=1)
    ax.plot([89, 89], [-42.5, 42.5], color="red", linewidth=1)
    # center circle
    center_circle = patches.Circle((0, 0), radius=15, fill=False, color="red", linewidth=0.8)
    ax.add_patch(center_circle)
    ax.set_xlim(-100, 100)
    ax.set_ylim(-42.5, 42.5)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return ax

