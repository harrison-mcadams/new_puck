"""Helpers to draw a schematic NHL rink with matplotlib.

The drawing is intentionally schematic (approximate measurements in feet) and
uses a rectangle + semicircular endcaps. The helper returns the matplotlib
`Axes` so callers can overlay additional marks (shots, heatmaps, traces).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

RINK_LENGTH = 200.0
RINK_WIDTH = 85.0
# approximate goal x placement used elsewhere (feet from centerline)
GOAL_X = 89.0


def draw_rink(ax=None, center_line=True, mirror: bool = False, show_goals: bool = True):
    """Draw a simplified NHL rink as a stadium shape: center rectangle + semicircular endcaps.

    Only the outward-facing semicircular halves are filled (using Wedge). If
    mirror=True, the left/right ends are flipped horizontally.

    If show_goals=True (default) draw schematic goal creases and goal posts.
    """
    R = RINK_WIDTH / 2.0
    half_length = RINK_LENGTH / 2.0
    straight_half = half_length - R  # half-length of the straight center section

    left_center_x = -straight_half
    right_center_x = straight_half
    if mirror:
        # flip horizontally
        left_center_x, right_center_x = -left_center_x, -right_center_x

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.5))

    # center rectangle (background fill)
    rect = patches.Rectangle((left_center_x, -R),  # lower-left corner
                             width=(right_center_x - left_center_x),
                             height=2 * R,
                             facecolor="white",
                             edgecolor="none",
                             zorder=0)
    ax.add_patch(rect)

    # semicircular endcaps (background fill)
    left_theta = (90, 270) if left_center_x < 0 else (-90, 90)
    right_theta = (90, 270) if right_center_x < 0 else (-90, 90)

    left_wedge = patches.Wedge((left_center_x, 0), R, theta1=left_theta[0], theta2=left_theta[1], facecolor="white", edgecolor="none", zorder=0)
    right_wedge = patches.Wedge((right_center_x, 0), R, theta1=right_theta[0], theta2=right_theta[1], facecolor="white", edgecolor="none", zorder=0)
    ax.add_patch(left_wedge)
    ax.add_patch(right_wedge)

    # Draw outlines: top/bottom straight edges
    # Thickness 1.2 matches curves. clip_on=False prevents axis limits from shaving the line width
    ax.plot([left_center_x, right_center_x], [R, R], color="black", linewidth=1.2, zorder=2, clip_on=False)
    ax.plot([left_center_x, right_center_x], [-R, -R], color="black", linewidth=1.2, zorder=2, clip_on=False)

    # semicircle outlines using Arc
    # clip_on=False prevents axis limits from shaving the line width
    left_arc = patches.Arc((left_center_x, 0), width=2 * R, height=2 * R, angle=0, theta1=left_theta[0], theta2=left_theta[1], edgecolor="black", linewidth=1.2, zorder=2, clip_on=False)
    right_arc = patches.Arc((right_center_x, 0), width=2 * R, height=2 * R, angle=0, theta1=right_theta[0], theta2=right_theta[1], edgecolor="black", linewidth=1.2, zorder=2, clip_on=False)
    ax.add_patch(left_arc)
    ax.add_patch(right_arc)

    # center line
    if center_line:
        ax.plot([0, 0], [-R, R], color="red", linewidth=1, zorder=3)

    # --- BLUE LINES ---
    # approximate blue-line x positions (half-distance between center and endcap): use ±25 ft
    blue_x = 25.0
    left_blue_x = -blue_x
    right_blue_x = blue_x
    if mirror:
        left_blue_x, right_blue_x = -left_blue_x, -right_blue_x
    # draw blue lines as filled thin rectangles for visible width
    blue_width = 1.0
    ax.add_patch(patches.Rectangle((left_blue_x - blue_width/2, -R), blue_width, 2 * R, facecolor='blue', alpha=0.6, edgecolor='none', zorder=2))
    ax.add_patch(patches.Rectangle((right_blue_x - blue_width/2, -R), blue_width, 2 * R, facecolor='blue', alpha=0.6, edgecolor='none', zorder=2))
    # thin outlines for blue lines
    ax.plot([left_blue_x, left_blue_x], [-R, R], color='blue', linewidth=2, zorder=3)
    ax.plot([right_blue_x, right_blue_x], [-R, R], color='blue', linewidth=2, zorder=3)

    # goal lines: clip vertical extent to the rink surface (straight section or semicircle)
    def _y_extent_at_x(x: float) -> float:
        """Return half-height (positive) of rink at given x coordinate."""
        # if within straight section horizontally, full R applies
        if left_center_x <= x <= right_center_x:
            return R
        # else determine which semicircle center is closer (left or right)
        # and compute y = sqrt(R^2 - dx^2) if within circle, otherwise 0
        if x < left_center_x:
            center = left_center_x
        else:
            center = right_center_x
        dx = abs(x - center)
        if dx > R:
            return 0.0
        return math.sqrt(max(0.0, R * R - dx * dx))

    # goal x positions (approx. official placement)
    left_goal_x = -89.0
    right_goal_x = 89.0
    if mirror:
        left_goal_x, right_goal_x = -left_goal_x, -right_goal_x

    left_y = _y_extent_at_x(left_goal_x)
    right_y = _y_extent_at_x(right_goal_x)

    # draw vertical goal line segments clipped to rink
    if left_y > 0:
        ax.plot([left_goal_x, left_goal_x], [-left_y, left_y], color="red", linewidth=1, zorder=3)
    if right_y > 0:
        ax.plot([right_goal_x, right_goal_x], [-right_y, right_y], color="red", linewidth=1, zorder=3)

    # --- draw schematic goal creases and posts ---
    if show_goals:
        # crease parameters (feet)
        crease_radius = 6.0
        crease_alpha = 0.45
        # lightskyblue matches typical nhl crease
        crease_color = "lightskyblue"
        
        # goal mouth half-width (6 ft total -> 3 ft half-width)
        goal_half_width = 3.0
        # post radius
        post_radius = 0.6
        # inward offset to draw the net-mouth line slightly inside the crease
        inward_offset = -1.0 if right_goal_x > 0 else 1.0

        # left goal crease and posts
        if left_y > 0:
            # original crease facing toward center: left goal (neg x) would be (-90,90)
            orig_theta1, orig_theta2 = (-90, 90) if left_goal_x < 0 else (90, 270)
            # use the original orientation (crease faces toward center)
            theta1, theta2 = orig_theta1, orig_theta2
            crease = patches.Wedge((left_goal_x, 0), crease_radius, theta1=theta1, theta2=theta2, facecolor=crease_color, alpha=crease_alpha, edgecolor='none', zorder=1)
            ax.add_patch(crease)
            # posts REMOVED as requested
            # ax.add_patch(patches.Circle((left_goal_x, goal_half_width), radius=post_radius, color='red', zorder=4))
            # ax.add_patch(patches.Circle((left_goal_x, -goal_half_width), radius=post_radius, color='red', zorder=4))
            # net-mouth line placed away from center (one unit further from center)
            mouth_x = left_goal_x + (-1.0 if left_goal_x < 0 else 1.0)
            ax.plot([mouth_x, mouth_x], [-goal_half_width, goal_half_width], color='red', linewidth=1.2, zorder=4)

        # right goal crease and posts
        if right_y > 0:
            # original crease facing toward center: right goal (pos x) would be (90,270)
            orig_theta1, orig_theta2 = (90, 270) if right_goal_x > 0 else (-90, 90)
            # use the original orientation (crease faces toward center)
            theta1, theta2 = orig_theta1, orig_theta2
            crease = patches.Wedge((right_goal_x, 0), crease_radius, theta1=theta1, theta2=theta2, facecolor=crease_color, alpha=crease_alpha, edgecolor='none', zorder=1)
            ax.add_patch(crease)
            # posts REMOVED as requested
            # ax.add_patch(patches.Circle((right_goal_x, goal_half_width), radius=post_radius, color='red', zorder=4))
            # ax.add_patch(patches.Circle((right_goal_x, -goal_half_width), radius=post_radius, color='red', zorder=4))
            # net-mouth line placed away from center (one unit further from center)
            mouth_x = right_goal_x + (1.0 if right_goal_x > 0 else -1.0)
            ax.plot([mouth_x, mouth_x], [-goal_half_width, goal_half_width], color='red', linewidth=1.2, zorder=4)

    # --- FACEOFF CIRCLES (offensive/defensive zone circles) ---
    # approximate positions for the four end-zone faceoff circles
    face_x = 69.0
    face_y = 22.0
    face_radius = 15.0
    # flip for mirror
    fx_left = -face_x
    fx_right = face_x
    if mirror:
        fx_left, fx_right = -fx_left, -fx_right
    for cx in (fx_left, fx_right):
        for cy in (face_y, -face_y):
            # circle outline
            ax.add_patch(patches.Circle((cx, cy), radius=face_radius, fill=False, edgecolor='blue', linewidth=1.2, zorder=2))
            # faceoff dot
            ax.add_patch(patches.Circle((cx, cy), radius=0.5, color='black', zorder=4))

    # center circle
    center_circle = patches.Circle((0, 0), radius=15, fill=False, color="red", linewidth=0.8, zorder=3)
    ax.add_patch(center_circle)

    ax.set_xlim(-half_length, half_length)
    ax.set_ylim(-R, R)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return ax


def rink_half_height_at_x(x: float) -> float:
    """Return half-height of the rink at coordinate x.

    This mirrors the internal helper used in draw_rink to determine how far
    from the center the rink extends vertically at horizontal position x.
    Useful for masking simulated points to the rink shape.
    """
    R = RINK_WIDTH / 2.0
    half_length = RINK_LENGTH / 2.0
    straight_half = half_length - R

    left_center_x = -straight_half
    right_center_x = straight_half

    # if within straight section horizontally, full R applies
    if left_center_x <= x <= right_center_x:
        return R

    # else determine which semicircle center is closer (left or right)
    if x < left_center_x:
        center = left_center_x
    else:
        center = right_center_x
    dx = abs(x - center)
    if dx > R:
        return 0.0
    return math.sqrt(max(0.0, R * R - dx * dx))


def rink_bounds():
    """Return (xmin, xmax, ymin, ymax) bounds of the rink surface in feet.

    xmin/xmax are along the length axis, ymin/ymax are half-widths along y.
    """
    R = RINK_WIDTH / 2.0
    half_length = RINK_LENGTH / 2.0
    return -half_length, half_length, -R, R


def rink_goal_xs(mirror: bool = False):
    """Return (left_goal_x, right_goal_x) coordinates.

    If mirror=True the values are flipped.
    """
    left_goal_x = -GOAL_X
    right_goal_x = GOAL_X
    if mirror:
        return -left_goal_x, -right_goal_x
    return left_goal_x, right_goal_x


def calculate_distance_and_angle(x: float, y: float, goal_x: float, goal_y: float = 0.0) -> tuple[float, float]:
    """
    Standard calculation for distance and angle to the goal.
    Angle is 0-360 degrees:
    - 0 at goalie's left (along goal-line vector)
    - Increases clockwise
    """
    distance = math.hypot(x - goal_x, y - goal_y)
    
    # Vector from goal center to shot:
    vx = x - goal_x
    vy = y - goal_y

    # Reference vector along goal line pointing toward goalie's left:
    # If goal_x < 0 (left goal), goalie faces +x so his left is +y.
    # If goal_x > 0 (right goal), goalie faces -x so his left is -y.
    if goal_x < 0:
        rx, ry = 0.0, 1.0
    else:
        rx, ry = 0.0, -1.0

    # Signed angle from ref r to vector v (CCW positive): atan2(cross, dot)
    cross = rx * vy - ry * vx
    dot = rx * vx + ry * vy
    angle_rad_ccw = math.atan2(cross, dot)

    # We want clockwise positive, so invert sign, convert to degrees
    angle_deg = (-math.degrees(angle_rad_ccw)) % 360.0
    
    return distance, angle_deg
