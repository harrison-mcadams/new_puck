
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
    shot_types = ['wrist', 'slap', 'snap', 'backhand', 'tip-in']
    shot_type_labels = {
        'wrist': 'Wrist Shot', 'slap': 'Slap Shot', 'snap': 'Snap Shot',
        'backhand': 'Backhand', 'tip-in': 'Tip-In'
    }
    
    combinations = list(itertools.product(roles, game_states, score_states.keys(), shot_types))
    print(f"Pre-computing {len(combinations)} combinations...")

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Current Block Prob", "Current xG Prob",
            "Block Delta (vs Baseline)", "xG Delta (vs Baseline)"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    for i, (role, gs, ss_label, st) in enumerate(combinations):
        if (i+1) % 20 == 0: print(f"  [{i+1}/{len(combinations)}]")
        
        df = grid_df_base.copy()
        df['shooter_role'] = role
        df['game_state'] = gs
        df['score_diff'] = score_states[ss_label]
        df['shot_type'] = st
        
        processed_df = data_pipeline.preprocess_features(
            df, is_training=False, apply_imputation=False,
            apply_arena_adjustments=False, apply_dithering=False, apply_filtering=False
        )
        
        block_probs = model.predict_proba_layer(processed_df, layer='block').reshape(xx.shape)
        xg_probs = model.predict_proba(processed_df)[:, 1].reshape(xx.shape)
        
        # Masking
        for r in range(xx.shape[0]):
            for c in range(xx.shape[1]):
                if not (abs(yy[r, c]) <= rink.rink_half_height_at_x(xx[r, c])):
                    block_probs[r, c] = np.nan
                    xg_probs[r, c] = np.nan

        scenario_id = f"{role}_{gs}_{ss_label}_{st.replace('-', '_')}"
        is_visible = (role == 'F' and gs == '5v5' and ss_label == 'Tied' and st == 'wrist')
        st_label = shot_type_labels[st]
        
        # Block Trace
        fig.add_trace(go.Heatmap(
            x=xs, y=ys, z=block_probs.tolist(),
            coloraxis="coloraxis1",
            name=scenario_id,
            visible=is_visible,
            hovertemplate=f"<b>{role} | {gs} | {ss_label}</b><br>Block Prob: %{{z:.4f}}<extra></extra>"
        ), row=1, col=1)
        
        # xG Trace
        fig.add_trace(go.Heatmap(
            x=xs, y=ys, z=xg_probs.tolist(),
            coloraxis="coloraxis2",
            name=scenario_id,
            visible=is_visible,
            hovertemplate=f"<b>{role} | {gs} | {ss_label}</b><br>xG Prob: %{{z:.4f}}<extra></extra>"
        ), row=1, col=2)

    # Delta Traces (Traces 2N and 2N+1)
    zeros = np.zeros_like(xx).tolist()
    fig.add_trace(go.Heatmap(
        x=xs, y=ys, z=zeros,
        coloraxis="coloraxis3",
        name="delta_block",
        visible=True,
        hovertemplate="Block Delta: %{z:.4f}<extra></extra>"
    ), row=2, col=1)

    fig.add_trace(go.Heatmap(
        x=xs, y=ys, z=zeros,
        coloraxis="coloraxis4",
        name="delta_xg",
        visible=True,
        hovertemplate="xG Delta: %{z:.4f}<extra></extra>"
    ), row=2, col=2)

    fig.update_layout(
        title=dict(text="Blocked Shot Comparison Dashboard", x=0.5, font=dict(size=24, color='white')),
        width=1600, height=1400,
        paper_bgcolor='#111',
        plot_bgcolor='#111',
        coloraxis1=dict(colorscale='Magma', cmin=0, cmax=1.0, colorbar=dict(title="Block Prob", x=0.45, y=0.82, len=0.35)),
        coloraxis2=dict(colorscale='Plasma', cmin=0, cmax=0.3, colorbar=dict(title="xG Prob", x=1.0, y=0.82, len=0.35)),
        coloraxis3=dict(colorscale='RdBu_r', cmin=-0.2, cmax=0.2, colorbar=dict(title="Block Δ", x=0.45, y=0.25, len=0.35)),
        coloraxis4=dict(colorscale='RdBu_r', cmin=-0.1, cmax=0.1, colorbar=dict(title="xG Δ", x=1.0, y=0.25, len=0.35)),
        shapes=(
            get_rink_shapes(xref='x', yref='y') + get_rink_shapes(xref='x2', yref='y2') +
            get_rink_shapes(xref='x3', yref='y3') + get_rink_shapes(xref='x4', yref='y4')
        )
    )

    fig.update_layout(
        xaxis=dict(range=[0, 100], showgrid=False, zeroline=False, color='white'),
        yaxis=dict(range=[-42.5, 42.5], showgrid=False, zeroline=False, color='white', scaleanchor="x", scaleratio=1),
        xaxis2=dict(range=[0, 100], showgrid=False, zeroline=False, color='white'),
        yaxis2=dict(range=[-42.5, 42.5], showgrid=False, zeroline=False, color='white', scaleanchor="x2", scaleratio=1),
        xaxis3=dict(range=[0, 100], showgrid=False, zeroline=False, color='white'),
        yaxis3=dict(range=[-42.5, 42.5], showgrid=False, zeroline=False, color='white', scaleanchor="x3", scaleratio=1),
        xaxis4=dict(range=[0, 100], showgrid=False, zeroline=False, color='white'),
        yaxis4=dict(range=[-42.5, 42.5], showgrid=False, zeroline=False, color='white', scaleanchor="x4", scaleratio=1)
    )

    # HTML UI
    def gen_options(items, labels=None):
        return "".join([f'<option value="{it}">{labels[it] if labels else it}</option>' for it in items])

    controls_html = f"""
    <div id="controls" style="display: flex; justify-content: center; align-items: center; gap: 20px; padding: 15px; background: #222; color: white; border-radius: 8px; margin-bottom: 5px; font-family: sans-serif; border: 1px solid #444;">
        <div><label><b>Role:</b></label> <select id="role-select" style="background:#444; color:white; border:none; padding:5px;">{gen_options(roles)}</select></div>
        <div><label><b>Game State:</b></label> <select id="gs-select" style="background:#444; color:white; border:none; padding:5px;">{gen_options(game_states)}</select></div>
        <div><label><b>Score State:</b></label> <select id="ss-select" style="background:#444; color:white; border:none; padding:5px;">{gen_options(score_states.keys())}</select></div>
        <div><label><b>Shot Type:</b></label> <select id="st-select" style="background:#444; color:white; border:none; padding:5px;">{gen_options(shot_types, shot_type_labels)}</select></div>
        <div style="border-left: 1px solid #555; padding-left: 20px; margin-left: 10px; display: flex; gap: 10px;">
            <button id="set-baseline" style="background: #28a745; color: white; border: none; padding: 8px 15px; border-radius: 4px; cursor: pointer; font-weight: bold;">Set Baseline</button>
            <button id="restore-baseline" style="background: #dc3545; color: white; border: none; padding: 8px 15px; border-radius: 4px; cursor: pointer; font-weight: bold;">Restore Baseline</button>
        </div>
        <div id="baseline-status" style="font-size: 0.85em; color: #888;">Baseline: None (Standard View)</div>
    </div>
    """

    js_code = """
    <script>
    let baselineBlock = null;
    let baselineXG = null;
    let baselineSettings = null;

    function updateViz() {
        const role = document.getElementById('role-select').value;
        const gs = document.getElementById('gs-select').value;
        const ss = document.getElementById('ss-select').value;
        const st = document.getElementById('st-select').value.replace(/-/g, '_');
        const targetID = role + '_' + gs + '_' + ss + '_' + st;
        
        const gd = document.getElementsByClassName('plotly-graph-div')[0];
        let currentBlock = null;
        let currentXG = null;
        const visibility = [];
        
        console.log("Searching for targetID: " + targetID);
        
        for (let i = 0; i < gd.data.length - 2; i++) {
            if (gd.data[i].name === targetID) {
                visibility.push(true);
                if (i % 2 === 0) currentBlock = gd.data[i].z;
                else currentXG = gd.data[i].z;
            } else {
                visibility.push(false);
            }
        }
        
        visibility.push(true); // Delta Block
        visibility.push(true); // Delta xG
        
        Plotly.restyle(gd, {visible: visibility});

        // Update Delta Rinks
        if (baselineBlock && currentBlock && baselineXG && currentXG) {
            console.log("Calculating Deltas...");
            const deltaBlock = currentBlock.map((row, i) => row.map((val, j) => {
                if (val === null || baselineBlock[i][j] === null) return null;
                return val - baselineBlock[i][j];
            }));
            const deltaXG = currentXG.map((row, i) => row.map((val, j) => {
                if (val === null || baselineXG[i][j] === null) return null;
                return val - baselineXG[i][j];
            }));
            
            Plotly.restyle(gd, {z: [deltaBlock]}, [gd.data.length - 2]);
            Plotly.restyle(gd, {z: [deltaXG]}, [gd.data.length - 1]);
        } else {
            console.log("No baseline or current data found for both layers. Zeroing deltas.");
            const zeroZ = (currentBlock || gd.data[0].z).map(row => row.map(() => 0));
            Plotly.restyle(gd, {z: [zeroZ]}, [gd.data.length - 2, gd.data.length - 1]);
        }
    }

    document.getElementById('set-baseline').onclick = () => {
        const role = document.getElementById('role-select').value;
        const gs = document.getElementById('gs-select').value;
        const ss = document.getElementById('ss-select').value;
        const st = document.getElementById('st-select').value.replace(/-/g, '_');
        const targetID = role + '_' + gs + '_' + ss + '_' + st;
        
        const gd = document.getElementsByClassName('plotly-graph-div')[0];
        for (let i = 0; i < gd.data.length - 2; i++) {
            if (gd.data[i].name === targetID) {
                if (i % 2 === 0) baselineBlock = gd.data[i].z;
                else baselineXG = gd.data[i].z;
            }
        }
        baselineSettings = {role, gs, ss, st};
        document.getElementById('baseline-status').innerText = `Baseline: ${role} | ${gs} | ${ss} | ${st}`;
        document.getElementById('baseline-status').style.color = "#28a745";
        updateViz();
    };

    document.getElementById('restore-baseline').onclick = () => {
        if (!baselineSettings) return;
        document.getElementById('role-select').value = baselineSettings.role;
        document.getElementById('gs-select').value = baselineSettings.gs;
        document.getElementById('ss-select').value = baselineSettings.ss;
        document.getElementById('st-select').value = baselineSettings.st.replace(/_/g, '-');
        updateViz();
    };

    ["role-select", "gs-select", "ss-select", "st-select"].forEach(id => {
        document.getElementById(id).onchange = updateViz;
    });
    </script>
    """

    plot_html = fig.to_html(include_plotlyjs='cdn', full_html=False)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"<!DOCTYPE html><html><body style='margin:0; background:#111; color:white;'>{controls_html}{plot_html}{js_code}</body></html>")
    print(f"2x2 Comparison Dashboard generated: {output_path}")

if __name__ == "__main__":
    main()
