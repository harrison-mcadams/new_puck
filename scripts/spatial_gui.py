import sys
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider, CheckButtons, Button
from pathlib import Path
import math

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from puck import fit_xgboost_nested
from puck.rink import draw_rink, rink_bounds, rink_half_height_at_x, rink_goal_xs, calculate_distance_and_angle

def main():
    # 1. Load Model
    model_path = Path('analysis/xgs/xg_model_nested_all.joblib')
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return
    
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    # 2. Setup Rink Grid
    x_res, y_res = 1.0, 1.0
    xmin, xmax, ymin, ymax = rink_bounds()
    gx = np.arange(xmin, xmax + x_res, x_res)
    gy = np.arange(ymin, ymax + y_res, y_res)
    XX, YY = np.meshgrid(gx, gy)
    
    # Mask outside rink
    mask = np.vectorize(rink_half_height_at_x)(XX) >= np.abs(YY)
    # Mask offensive zone only (x > 0) to save computation and focus on shots
    mask &= (XX > 0)
    
    left_goal_x, right_goal_x = rink_goal_xs()
    goal_x = right_goal_x # We assume we attack the right goal for visualization
    
    # Precompute distances and angles for grid points
    flat_pts = []
    indices = []
    for i in range(XX.shape[0]):
        for j in range(XX.shape[1]):
            if mask[i, j]:
                x, y = XX[i, j], YY[i, j]
                dist, angle = calculate_distance_and_angle(x, y, goal_x)
                flat_pts.append({'distance': dist, 'angle_deg': angle})
                indices.append((i, j))
    
    # 3. GUI Setup
    fig = plt.figure(figsize=(16, 10))
    # Main plot area
    ax_map = plt.axes([0.3, 0.25, 0.65, 0.65])
    draw_rink(ax=ax_map)
    ax_map.set_title("Interactive xG Map", fontsize=16)
    
    extent = [gx[0], gx[-1], gy[0], gy[-1]]
    im = ax_map.imshow(np.zeros_like(XX), extent=extent, origin='lower', cmap='viridis', vmin=0, vmax=0.3, alpha=0.8, zorder=1)
    cbar = fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
    cbar.set_label('xG Probability')

    # Current State Dictionary
    state = {
        'game_state': '5v5',
        'shot_type': 'wrist',
        'shoots_catches': 'L',
        'last_event_type': 'Faceoff',
        'score_diff': 0,
        'period_number': 1,
        'total_time_elapsed_s': 600,
        'time_elapsed_in_period_s': 600,
        'rebound_angle_change': 0.0,
        'rebound_time_diff': 0.0,
        'last_event_time_diff': 10.0,
        'is_rebound': 0,
        'is_rush': 0,
        'diff_mode': False,
        'baseline_probs': None
    }

    # --- WIDGETS ---
    # Layout constants
    col1_x = 0.02
    col2_x = 0.15
    widget_width = 0.12
    
    # Radio Buttons
    ax_shot = plt.axes([col1_x, 0.75, widget_width, 0.2])
    ax_shot.set_title("Shot Type")
    radio_shot = RadioButtons(ax_shot, fit_xgboost_nested.VOCAB_SHOT_TYPE)
    
    ax_gs = plt.axes([col1_x, 0.55, widget_width, 0.15])
    ax_gs.set_title("Game State")
    radio_gs = RadioButtons(ax_gs, ['5v5', '5v4', '4v5', '4v4', '3v3'])
    
    ax_hand = plt.axes([col1_x, 0.40, widget_width, 0.1])
    ax_hand.set_title("Shoots")
    radio_hand = RadioButtons(ax_hand, ['L', 'R'])
    
    ax_last = plt.axes([col1_x, 0.1, widget_width, 0.25])
    ax_last.set_title("Last Event")
    radio_last = RadioButtons(ax_last, ['Faceoff', 'Hit', 'Giveaway', 'Shot', 'Blocked Shot'])

    # Sliders
    ax_score = plt.axes([col2_x, 0.9, widget_width, 0.03])
    slide_score = Slider(ax_score, 'Score Diff', -5, 5, valinit=0, valfmt='%d')
    
    ax_period = plt.axes([col2_x, 0.85, widget_width, 0.03])
    slide_period = Slider(ax_period, 'Period', 1, 4, valinit=1, valfmt='%d')
    
    ax_time = plt.axes([col2_x, 0.80, widget_width, 0.03])
    slide_time = Slider(ax_time, 'Time (s)', 0, 3600, valinit=600)
    
    ax_rebound_angle = plt.axes([col2_x, 0.70, widget_width, 0.03])
    slide_rebound_angle = Slider(ax_rebound_angle, 'Reb Angle', 0, 180, valinit=0)
    
    ax_rebound_time = plt.axes([col2_x, 0.65, widget_width, 0.03])
    slide_rebound_time = Slider(ax_rebound_time, 'Reb Time', 0, 10, valinit=0)
    
    ax_last_time = plt.axes([col2_x, 0.55, widget_width, 0.03])
    slide_last_time = Slider(ax_last_time, 'Last Event (s)', 0, 60, valinit=10)

    # Checkbuttons
    ax_check = plt.axes([col2_x, 0.4, widget_width, 0.1])
    check = CheckButtons(ax_check, ['Is Rebound', 'Is Rush', 'Diff Mode'], [False, False, False])

    # Baseline Button
    ax_base = plt.axes([col2_x, 0.3, widget_width, 0.05])
    btn_base = Button(ax_base, 'Set Baseline')

    def build_feature_df():
        # Build DataFrame for all points
        data = {f: [0.0] * len(flat_pts) for f in model.features}
        for k, p in enumerate(flat_pts):
            data['distance'][k] = p['distance']
            data['angle_deg'][k] = p['angle_deg']
            # Situational
            data['game_state'][k] = state['game_state']
            data['shot_type'][k] = state['shot_type']
            data['shoots_catches'][k] = state['shoots_catches']
            data['last_event_type'][k] = state['last_event_type']
            data['score_diff'][k] = state['score_diff']
            data['period_number'][k] = state['period_number']
            data['total_time_elapsed_s'][k] = state['total_time_elapsed_s']
            data['time_elapsed_in_period_s'][k] = state['total_time_elapsed_s'] % 1200
            data['rebound_angle_change'][k] = state['rebound_angle_change']
            data['rebound_time_diff'][k] = state['rebound_time_diff']
            data['last_event_time_diff'][k] = state['last_event_time_diff']
            data['is_rebound'][k] = state['is_rebound']
            data['is_rush'][k] = state['is_rush']
            
        return pd.DataFrame(data)

    def update(_):
        # Read from widgets
        state['shot_type'] = radio_shot.value_selected
        state['game_state'] = radio_gs.value_selected
        state['shoots_catches'] = radio_hand.value_selected
        state['last_event_type'] = radio_last.value_selected
        
        state['score_diff'] = int(slide_score.val)
        state['period_number'] = int(slide_period.val)
        state['total_time_elapsed_s'] = slide_time.val
        state['rebound_angle_change'] = slide_rebound_angle.val
        state['rebound_time_diff'] = slide_rebound_time.val
        state['last_event_time_diff'] = slide_last_time.val
        
        checks = check.get_status()
        state['is_rebound'] = int(checks[0])
        state['is_rush'] = int(checks[1])
        state['diff_mode'] = checks[2]
        
        # Predict
        df_feat = build_feature_df()
        raw_probs = model.predict_proba(df_feat)
        probs = raw_probs[:, 1].ravel()
        
        heat = np.full(XX.shape, np.nan)
        for (i, j), p in zip(indices, probs):
            heat[i, j] = p
            
        if state['diff_mode'] and state['baseline_probs'] is not None:
            # Force 1D comparison
            baseline = state['baseline_probs'].ravel()
            diff_values = probs - baseline
            
            diff_heat = np.full(XX.shape, np.nan)
            for (i, j), d in zip(indices, diff_values):
                diff_heat[i, j] = d
            im.set_data(diff_heat)
            im.set_clim(-0.05, 0.05) # Tighter scale for sensitivity
            im.set_cmap('RdBu_r')
            ax_map.set_title(f"Differential: Current vs Baseline", color='red')
        else:
            im.set_data(heat)
            im.set_clim(0, 0.3)
            im.set_cmap('viridis')
            ax_map.set_title(f"xG: {state['shot_type']} | {state['game_state']}", color='black')
            
        fig.canvas.draw_idle()

    def set_baseline(event):
        df_feat = build_feature_df()
        state['baseline_probs'] = model.predict_proba(df_feat)[:, 1]
        print("Baseline captured.")

    # Attach events
    radio_shot.on_clicked(update)
    radio_gs.on_clicked(update)
    radio_hand.on_clicked(update)
    radio_last.on_clicked(update)
    slide_score.on_changed(update)
    slide_period.on_changed(update)
    slide_time.on_changed(update)
    slide_rebound_angle.on_changed(update)
    slide_rebound_time.on_changed(update)
    slide_last_time.on_changed(update)
    check.on_clicked(update)
    btn_base.on_clicked(set_baseline)

    # Initial Draw
    update(None)
    plt.show()

if __name__ == "__main__":
    main()
