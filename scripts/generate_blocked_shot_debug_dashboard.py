
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
    
    # Inject Priors for Marginalization (if missing from old model)
    if not getattr(model, 'shot_type_priors_', None):
        print("Injecting default shot_type_priors_ for marginalization...")
        model.shot_type_priors_ = {
            'wrist': 0.55,
            'snap': 0.15, 
            'slap': 0.15,
            'backhand': 0.10,
            'tip-in': 0.05
        }
    
    # Grid Setup
    xs = np.linspace(0, 100, 100)
    ys = np.linspace(-42.5, 42.5, 85)
    xx, yy = np.meshgrid(xs, ys)
    grid_df_base = pd.DataFrame({'x': xx.ravel(), 'y': yy.ravel()})
    
    # Updated Defaults (Overwritten by Combinations)
    grid_df_base['event'] = 'shot-on-goal'
    grid_df_base['period_number'] = 2
    grid_df_base['time_elapsed_in_period_s'] = 600.0
    grid_df_base['total_time_elapsed_s'] = 1800.0
    grid_df_base['last_event_time_diff'] = 10.0
    grid_df_base['period_time_type'] = 'elapsed'
    grid_df_base['home_team_defending_side'] = 'left' 
    grid_df_base['player_name'] = 'Average Joe'
    grid_df_base['team_abbrev'] = 'AVG'
    grid_df_base['home_abb'] = 'AVG'
    grid_df_base['away_abb'] = 'OPP'

    # Filter Dimensions (REDUCED for manageable file size)
    roles = ['F', 'D']
    game_states = ['5v5']  # Only 5v5 for now (can expand later)
    score_states = {'Tied': 0, 'Leading': 1, 'Trailing': -1}
    shot_types = ['wrist', 'slap']  # 2 most common, reduces from 5
    
    # NEW FEATURES (REDUCED)
    handedness = ['L', 'R']
    rush_opts = [0]  # Only No-Rush for now
    rebound_opts = [0]  # Only No-Rebound for now
    prior_events = ['Faceoff']  # Only 1 prior event for now

    shot_type_labels = {
        'wrist': 'Wrist', 'slap': 'Slap', 'snap': 'Snap',
        'backhand': 'Backhand', 'tip-in': 'Tip-In'
    }
    
    # Combinations: Role(2) * GS(1) * Score(3) * Type(2) * Hand(2) * Rush(1) * Reb(1) * Event(1)
    # Total: 2*1*3*2*2*1*1*1 = 24 combinations. Manageable!
    
    combinations = list(itertools.product(roles, game_states, score_states.keys(), shot_types, handedness, rush_opts, rebound_opts, prior_events))
    print(f"Pre-computing {len(combinations)} combinations... (This might take a minute)")

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Current Block Prob", "Current xG Prob",
            "Block Delta (vs Baseline)", "xG Delta (vs Baseline)"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # Marginalization Grids (Nuisance Features)
    # We will average predictions over these states
    periods = [1, 2, 3]
    times = [300.0, 900.0] # Early, Late (Mid is implicitly covered by avg)
    time_diffs = [10.0] # Keep fixed to save compute (3x2x1 = 6x multiplier)
    
    marg_combos = list(itertools.product(periods, times, time_diffs))
    
    for i, (role, gs, ss_label, st, hand, is_rush, is_rebound, prev_evt) in enumerate(combinations):
        if (i+1) % 100 == 0: print(f"  [{i+1}/{len(combinations)}]", end='\r')
        
        # Stack copies of base grid for each marginalization combo
        dfs = []
        for (p, t, td) in marg_combos:
            _df = grid_df_base.copy()
            _df['shooter_role'] = role
            _df['game_state'] = gs
            _df['score_diff'] = score_states[ss_label]
            _df['shot_type'] = st
            _df['shoots_catches'] = hand
            _df['is_rush'] = is_rush
            _df['is_rebound'] = is_rebound
            _df['last_event_type'] = prev_evt
            
            # Marginalized features
            _df['period_number'] = p
            _df['time_elapsed_in_period_s'] = t
            _df['last_event_time_diff'] = td
            
            dfs.append(_df)
            
        full_df = pd.concat(dfs, ignore_index=True)
        
        # Batch Preprocess
        processed_df = data_pipeline.preprocess_features(
            full_df, is_training=False, apply_imputation=False,
            apply_arena_adjustments=False, apply_dithering=False, apply_filtering=False
        )
        
        # Batch Predict (Shape: [N_scenarios * N_grid, 1])
        all_block_probs = model.predict_proba_layer(processed_df, layer='block')
        all_xg_probs = model.predict_proba(processed_df)[:, 1]
        
        # Reshape to [N_scenarios, N_grid] and Average
        n_grid = len(grid_df_base)
        n_scenarios = len(marg_combos)
        
        block_probs_mat = all_block_probs.reshape(n_scenarios, n_grid)
        xg_probs_mat = all_xg_probs.reshape(n_scenarios, n_grid)
        
        avg_block = block_probs_mat.mean(axis=0).reshape(xx.shape)
        avg_xg = xg_probs_mat.mean(axis=0).reshape(xx.shape)
        
        # Masking (Same logic)
        for r in range(xx.shape[0]):
            for c in range(xx.shape[1]):
                if not (abs(yy[r, c]) <= rink.rink_half_height_at_x(xx[r, c])):
                    avg_block[r, c] = np.nan
                    avg_xg[r, c] = np.nan

        # Scenario ID (Underscore separated)
        evt_key = prev_evt.replace(" ", "")
        scenario_id = f"{role}_{gs}_{ss_label}_{st}_{hand}_{is_rush}_{is_rebound}_{evt_key}"
        
        # Default Visibility
        is_visible = (role == 'F' and gs == '5v5' and ss_label == 'Tied' and st == 'wrist' and 
                      hand == 'L' and is_rush == 0 and is_rebound == 0 and prev_evt == 'Faceoff')

        # Block Trace
        fig.add_trace(go.Heatmap(
            x=xs, y=ys, z=avg_block.tolist(),
            coloraxis="coloraxis1",
            name=scenario_id,
            visible=is_visible,
            hovertemplate=f"<b>{role}|{gs}|{hand}</b><br>Block: %{{z:.2f}}<extra></extra>"
        ), row=1, col=1)
        
        # xG Trace
        fig.add_trace(go.Heatmap(
            x=xs, y=ys, z=avg_xg.tolist(),
            coloraxis="coloraxis2",
            name=scenario_id,
            visible=is_visible,
            hovertemplate=f"xG: %{{z:.2f}}<extra></extra>"
        ), row=1, col=2)

    # Delta Traces (Traces 2N and 2N+1)
    zeros = np.zeros_like(xx).tolist()
    fig.add_trace(go.Heatmap(
        x=xs, y=ys, z=zeros,
        coloraxis="coloraxis3",
        name="delta_block",
        visible=True,
        hovertemplate="Block Delta: %{z:.3f}<extra></extra>"
    ), row=2, col=1)

    fig.add_trace(go.Heatmap(
        x=xs, y=ys, z=zeros,
        coloraxis="coloraxis4",
        name="delta_xg",
        visible=True,
        hovertemplate="xG Delta: %{z:.3f}<extra></extra>"
    ), row=2, col=2)

    fig.update_layout(
        title=dict(text="Blocked Shot Model Explorer (Marginalized)", x=0.5, font=dict(size=24, color='white')),
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
    
    # Values for boolean/cleaned selects
    rush_labels = {0: 'No', 1: 'Yes'}
    rebound_labels = {0: 'No', 1: 'Yes'}
    evt_vals = [e.replace(" ", "") for e in prior_events]
    evt_labels = {e.replace(" ", ""): e for e in prior_events}

    controls_html = f"""
    <div id="controls" style="display: flex; flex-wrap: wrap; justify-content: center; gap: 15px; padding: 15px; background: #222; color: white; border-radius: 8px; margin-bottom: 5px; font-family: sans-serif; border: 1px solid #444;">
        <div><label><b>Role:</b></label> <select id="role-select" class="ctrl">{gen_options(roles)}</select></div>
        <div><label><b>Game State:</b></label> <select id="gs-select" class="ctrl">{gen_options(game_states)}</select></div>
        <div><label><b>Score:</b></label> <select id="ss-select" class="ctrl">{gen_options(score_states.keys())}</select></div>
        <div><label><b>Shot:</b></label> <select id="st-select" class="ctrl">{gen_options(shot_types, shot_type_labels)}</select></div>
        
        <div style="border-left: 1px solid #555; padding-left: 15px;"></div>
        
        <div><label><b>Hand:</b></label> <select id="hand-select" class="ctrl">{gen_options(handedness)}</select></div>
        <div><label><b>Rush:</b></label> <select id="rush-select" class="ctrl">{gen_options(rush_opts, rush_labels)}</select></div>
        <div><label><b>Rebound:</b></label> <select id="reb-select" class="ctrl">{gen_options(rebound_opts, rebound_labels)}</select></div>
        <div><label><b>Prev Event:</b></label> <select id="evt-select" class="ctrl">{gen_options(evt_vals, evt_labels)}</select></div>

        <div style="border-left: 1px solid #555; padding-left: 15px; display: flex; gap: 10px;">
            <button id="set-baseline" style="background: #28a745; color: white; border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer; font-weight: bold;">Set Baseline</button>
            <button id="restore-baseline" style="background: #dc3545; color: white; border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer; font-weight: bold;">Restore</button>
        </div>
    </div>
    <div style="text-align:center; color:#888; font-family:sans-serif; margin-bottom:10px;" id="baseline-status">Baseline: None</div>
    <style> .ctrl {{ background:#444; color:white; border:none; padding:5px; border-radius:3px; }} </style>
    """

    js_code = """
    <script>
    let baselineBlock = null;
    let baselineXG = null;
    let baselineSettings = null;

    function getTargetID() {
        const role = document.getElementById('role-select').value;
        const gs = document.getElementById('gs-select').value;
        const ss = document.getElementById('ss-select').value;
        const st = document.getElementById('st-select').value;
        const hand = document.getElementById('hand-select').value;
        const rush = document.getElementById('rush-select').value;
        const reb = document.getElementById('reb-select').value;
        const evt = document.getElementById('evt-select').value;

        // Order: Role_GS_Score_Type_Hand_Rush_Reb_Event
        return [role, gs, ss, st, hand, rush, reb, evt].join('_');
    }

    function updateViz() {
        const targetID = getTargetID();
        const gd = document.getElementsByClassName('plotly-graph-div')[0];
        
        if (!gd || !gd.data) {
            console.warn("Graph not ready yet.");
            return;
        }

        let currentBlock = null;
        let currentXG = null;
        
        // Construct visibility array (very long!)
        // Traces are grouped in pairs (Block, xG)
        // Last 2 traces are Deltas
        
        const numScenarios = (gd.data.length - 2) / 2;
        const visibility = new Array(gd.data.length).fill(false);
        
        // Find Index
        // Optimization: We could compute index math if sorted, but search is safe
        let found = false;
        
        for (let i = 0; i < gd.data.length - 2; i += 2) {
            if (gd.data[i].name === targetID) {
                visibility[i] = true;     // Block
                visibility[i+1] = true;   // xG
                currentBlock = gd.data[i].z;
                currentXG = gd.data[i+1].z;
                found = true;
                break;
            }
        }
        
        if (!found) console.warn("Scenario not found: " + targetID);
        
        // Always show deltas
        visibility[gd.data.length - 2] = true; 
        visibility[gd.data.length - 1] = true;
        
        Plotly.restyle(gd, {visible: visibility});

        // Update Delta Rinks
        if (baselineBlock && currentBlock && baselineXG && currentXG) {
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
            const zeroZ = (currentBlock || gd.data[0].z).map(row => row.map(() => 0));
            Plotly.restyle(gd, {z: [zeroZ]}, [gd.data.length - 2, gd.data.length - 1]);
        }
    }

    document.getElementById('set-baseline').onclick = () => {
        const targetID = getTargetID();
        const gd = document.getElementsByClassName('plotly-graph-div')[0];
        if (!gd || !gd.data) return;
        
        for (let i = 0; i < gd.data.length - 2; i += 2) {
            if (gd.data[i].name === targetID) {
                baselineBlock = gd.data[i].z;
                baselineXG = gd.data[i+1].z;
                break;
            }
        }
        
        // Save current controls state
        baselineSettings = {
            role: document.getElementById('role-select').value,
            gs: document.getElementById('gs-select').value,
            ss: document.getElementById('ss-select').value,
            st: document.getElementById('st-select').value,
            hand: document.getElementById('hand-select').value,
            rush: document.getElementById('rush-select').value,
            reb: document.getElementById('reb-select').value,
            evt: document.getElementById('evt-select').value
        };
        
        const txt = Object.values(baselineSettings).join("|");
        const el = document.getElementById('baseline-status');
        el.innerText = "Baseline: " + txt;
        el.style.color = "#28a745";
        updateViz();
    };

    document.getElementById('restore-baseline').onclick = () => {
        if (!baselineSettings) return;
        document.getElementById('role-select').value = baselineSettings.role;
        document.getElementById('gs-select').value = baselineSettings.gs;
        document.getElementById('ss-select').value = baselineSettings.ss;
        document.getElementById('st-select').value = baselineSettings.st;
        document.getElementById('hand-select').value = baselineSettings.hand;
        document.getElementById('rush-select').value = baselineSettings.rush;
        document.getElementById('reb-select').value = baselineSettings.reb;
        document.getElementById('evt-select').value = baselineSettings.evt;
        updateViz();
    };

    const selects = document.getElementsByClassName('ctrl');
    for (let s of selects) {
        s.onchange = updateViz;
    }
    
    // Force initialize state on load to ensure match
    window.onload = function() {
        console.log("Dashboard loaded. Waiting for Plotly init...");
        const check = setInterval(() => {
            const gd = document.getElementsByClassName('plotly-graph-div')[0];
            if (gd && gd.data && gd.data.length > 0) {
                console.log("Plotly ready. Initializing viz.");
                clearInterval(check);
                updateViz();
            } else {
                console.log("Waiting for graph data...");
            }
        }, 200);
    };
    </script>
    """
    
    plot_html = fig.to_html(include_plotlyjs='cdn', full_html=False)
    # Increase config load to handle large number of traces if needed, though mostly standard
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"<!DOCTYPE html><html><head><meta charset='utf-8'/></head><body style='margin:0; background:#111; color:white;'>{controls_html}{plot_html}{js_code}</body></html>")
    print(f"2x2 Comparison Dashboard generated: {output_path}")

if __name__ == "__main__":
    main()
