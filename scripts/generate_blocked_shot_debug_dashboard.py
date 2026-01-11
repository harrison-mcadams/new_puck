
import sys
import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import data_pipeline, rink
from puck.fit_xgboost_nested import XGBNestedXGClassifier

def get_rink_shapes(xref='x', yref='y'):
    """Full rink shapes for offensive zone."""
    shapes = []
    line_color = "rgba(0, 0, 0, 0.3)"
    red_line_color = "rgba(255, 0, 0, 0.3)"
    blue_line_color = "rgba(0, 0, 255, 0.3)"
    
    shapes.append(dict(type="line", x0=0, y0=42.5, x1=100, y1=42.5, xref=xref, yref=yref, line=dict(color=line_color, width=2)))
    shapes.append(dict(type="line", x0=0, y0=-42.5, x1=100, y1=-42.5, xref=xref, yref=yref, line=dict(color=line_color, width=2)))
    shapes.append(dict(type="line", x0=100, y0=-42.5, x1=100, y1=42.5, xref=xref, yref=yref, line=dict(color=line_color, width=2)))
    shapes.append(dict(type="line", x0=0, y0=-42.5, x1=0, y1=42.5, xref=xref, yref=yref, line=dict(color=red_line_color, width=2)))
    shapes.append(dict(type="line", x0=25, y0=-42.5, x1=25, y1=42.5, xref=xref, yref=yref, line=dict(color=blue_line_color, width=2)))
    shapes.append(dict(type="line", x0=89, y0=-42.5, x1=89, y1=42.5, xref=xref, yref=yref, line=dict(color=red_line_color, width=1)))
    shapes.append(dict(type="circle", x0=89-4, y0=-4, x1=89+4, y1=4, xref=xref, yref=yref, line=dict(color=red_line_color, width=1)))
    for y_center in [22, -22]:
        shapes.append(dict(type="circle", x0=69-15, y0=y_center-15, x1=69+15, y1=y_center+15, xref=xref, yref=yref, line=dict(color=red_line_color, width=1)))
    return shapes

def main():
    model_path = "analysis/xgs/xg_model_nested.joblib"
    output_path = "analysis/blocked_shot_debug_dashboard.html"
    
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    model = joblib.load(model_path)
    
    # Grid Setup
    xs = np.linspace(0, 100, 100)
    ys = np.linspace(-42.5, 42.5, 85)
    xx, yy = np.meshgrid(xs, ys)
    grid_df_base = pd.DataFrame({'x': xx.ravel(), 'y': yy.ravel()})
    
    # Static Defaults
    grid_df_base['event'] = 'shot-on-goal'
    grid_df_base['is_rebound'] = 0
    grid_df_base['is_rush'] = 0
    grid_df_base['period_number'] = 2
    grid_df_base['time_elapsed_in_period_s'] = 600.0
    grid_df_base['total_time_elapsed_s'] = 1800.0
    grid_df_base['last_event_type'] = 'Faceoff'
    grid_df_base['last_event_time_diff'] = 10.0
    grid_df_base['shoots_catches'] = 'L'
    grid_df_base['period_time_type'] = 'elapsed'
    grid_df_base['home_team_defending_side'] = 'left' 
    grid_df_base['player_name'] = 'Average Joe'
    grid_df_base['team_abbrev'] = 'AVG'
    grid_df_base['home_abb'] = 'AVG'
    grid_df_base['away_abb'] = 'OPP'

    # Filter Dimensions
    roles = ['F', 'D']
    game_states = ['5v5', '5v4', '4v5']
    score_states = {'Tied': 0, 'Leading': 2, 'Trailing': -2}
    
    # Standardized shot type values matching VOCAB_SHOT_TYPE
    shot_types = ['wrist', 'slap', 'snap', 'backhand', 'tip-in']
    # Mapping for user-friendly display labels
    shot_type_labels = {
        'wrist': 'Wrist Shot',
        'slap': 'Slap Shot',
        'snap': 'Snap Shot',
        'backhand': 'Backhand',
        'tip-in': 'Tip-In'
    }
    
    combinations = list(itertools.product(roles, game_states, score_states.keys(), shot_types))
    print(f"Pre-computing {len(combinations)} combinations...")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Block Probability", "Expected Goals (xG)"),
        horizontal_spacing=0.1
    )

    # We will use "customdata" to store the scenario string for JS filtering
    for i, (role, gs, ss_label, st) in enumerate(combinations):
        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{len(combinations)}]")
        
        df = grid_df_base.copy()
        df['shooter_role'] = role
        df['game_state'] = gs
        df['score_diff'] = score_states[ss_label]
        df['shot_type'] = st
        
        processed_df = data_pipeline.preprocess_features(
            df, is_training=False, apply_imputation=False,
            apply_arena_adjustments=False, apply_dithering=False, apply_filtering=False
        )
        
        # Predictions
        block_probs = model.predict_proba_layer(processed_df, layer='block').reshape(xx.shape)
        xg_probs = model.predict_proba(processed_df)[:, 1].reshape(xx.shape)
        
        # Masking
        for r in range(xx.shape[0]):
            for c in range(xx.shape[1]):
                if not (abs(yy[r, c]) <= rink.rink_half_height_at_x(xx[r, c])):
                    block_probs[r, c] = np.nan
                    xg_probs[r, c] = np.nan

        scenario_id = f"{role}_{gs}_{ss_label}_{st.replace('-', '_')}"
        
        # Visibility: Default to F | 5v5 | Tied | wrist
        is_visible = (role == 'F' and gs == '5v5' and ss_label == 'Tied' and st == 'wrist')
        st_label = shot_type_labels[st]
        
        # Block Trace
        fig.add_trace(go.Heatmap(
            x=xs, y=ys, z=block_probs,
            coloraxis="coloraxis1",
            name=scenario_id,
            visible=is_visible,
            # Store metadata in customdata for JS to read
            customdata=[[scenario_id] * len(xs)] * len(ys),
            hovertemplate=f"<b>{role} | {gs} | {ss_label} | {st_label}</b><br>X: %{{x:.1f}}<br>Y: %{{y:.1f}}<br>Block Prob: %{{z:.4f}}<extra></extra>"
        ), row=1, col=1)
        
        # xG Trace
        fig.add_trace(go.Heatmap(
            x=xs, y=ys, z=xg_probs,
            coloraxis="coloraxis2",
            name=scenario_id,
            visible=is_visible,
            customdata=[[scenario_id] * len(xs)] * len(ys),
            hovertemplate=f"<b>{role} | {gs} | {ss_label} | {st_label}</b><br>X: %{{x:.1f}}<br>Y: %{{y:.1f}}<br>xG Prob: %{{z:.4f}}<extra></extra>"
        ), row=1, col=2)

    fig.update_layout(
        title=dict(text="Blocked Shot & xG Debug Dashboard", x=0.5, font=dict(size=24)),
        width=1800, height=850,
        coloraxis1=dict(colorscale='Magma', cmin=0, cmax=1.0, colorbar=dict(title="Block Prob", x=0.45)),
        coloraxis2=dict(colorscale='Plasma', cmin=0, cmax=0.3, colorbar=dict(title="xG Prob", x=1.0)),
        shapes=get_rink_shapes(xref='x', yref='y') + get_rink_shapes(xref='x2', yref='y2')
    )

    fig.update_layout(
        xaxis=dict(range=[0, 100], showgrid=False, zeroline=False),
        yaxis=dict(range=[-42.5, 42.5], showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1),
        xaxis2=dict(range=[0, 100], showgrid=False, zeroline=False),
        yaxis2=dict(range=[-42.5, 42.5], showgrid=False, zeroline=False, scaleanchor="x2", scaleratio=1)
    )

    # --- HTML / JS Injection ---
    
    # Generate the select options
    def gen_options(items, labels=None):
        if labels:
            return "".join([f'<option value="{it}">{labels[it]}</option>' for it in items])
        return "".join([f'<option value="{it}">{it}</option>' for it in items])

    controls_html = f"""
    <div id="controls" style="display: flex; justify-content: center; gap: 20px; padding: 10px; background: #f0f0f0; border-radius: 8px; margin-bottom: 10px; font-family: sans-serif;">
        <div>
            <label><b>Role:</b></label>
            <select id="role-select">{gen_options(roles)}</select>
        </div>
        <div>
            <label><b>Game State:</b></label>
            <select id="gs-select">{gen_options(game_states)}</select>
        </div>
        <div>
            <label><b>Score State:</b></label>
            <select id="ss-select">{gen_options(score_states.keys())}</select>
        </div>
        <div>
            <label><b>Shot Type:</b></label>
            <select id="st-select">{gen_options(shot_types, shot_type_labels)}</select>
        </div>
        <div style="font-size: 0.8em; color: #666; align-self: center;">
            <i>Showing 90 pre-computed scenarios</i>
        </div>
    </div>
    """

    js_code = """
    <script>
    function updateVisibility() {
        var role = document.getElementById('role-select').value;
        var gs = document.getElementById('gs-select').value;
        var ss = document.getElementById('ss-select').value;
        var st = document.getElementById('st-select').value.replace(/-/g, '_');
        
        var targetID = role + '_' + gs + '_' + ss + '_' + st;
        console.log("Switching to: " + targetID);
        
        var gd = document.getElementsByClassName('plotly-graph-div')[0];
        var update = {visible: []};
        
        for (var i = 0; i < gd.data.length; i++) {
            if (gd.data[i].name === targetID) {
                update.visible.push(true);
            } else {
                update.visible.push(false);
            }
        }
        
        Plotly.restyle(gd, update);
    }

    document.getElementById('role-select').addEventListener('change', updateVisibility);
    document.getElementById('gs-select').addEventListener('change', updateVisibility);
    document.getElementById('ss-select').addEventListener('change', updateVisibility);
    document.getElementById('st-select').addEventListener('change', updateVisibility);
    </script>
    """

    print(f"Writing to {output_path}...")
    
    # Write Plotly as HTML div and wrap with our custom UI
    plot_html = fig.to_html(include_plotlyjs='cdn', full_html=False)
    
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head><title>Blocked Shot & xG Debug Dashboard</title></head>
    <body style="margin: 20px;">
        {controls_html}
        {plot_html}
        {js_code}
    </body>
    </html>
    """
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_html)
        
    print("Dashboard generated successfully.")

if __name__ == "__main__":
    main()
