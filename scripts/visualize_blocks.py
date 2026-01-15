import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

def main():
    input_path = 'analysis/flyers_debug_2025020736.csv'
    output_path = 'analysis/blocked_shots_viz.png'
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    df = pd.read_csv(input_path)
    
    # Filter for blocked shots
    df_blocks = df[df['event'] == 'blocked-shot'].copy()
    
    if df_blocks.empty:
        print("No blocked shots found to visualize.")
        return

    # Check required columns
    required = ['x', 'y', 'block_x', 'block_y', 'team_id']
    if not all(col in df_blocks.columns for col in required):
        print(f"Missing columns. Found: {df_blocks.columns}")
        return
        
    teams = df_blocks['team_id'].unique()
    
    # Setup Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
    if len(teams) < 2:
        axes = [axes] # Handle single team case
        
    for i, team in enumerate(teams):
        ax = axes[i] if i < len(axes) else axes[0]
        subset = df_blocks[df_blocks['team_id'] == team]
        team_name = str(team) # In a real app we'd map ID to Name
        
        # Draw Rink Outline (Simplified)
        ax.set_xlim(-100, 100)
        ax.set_ylim(-42.5, 42.5)
        ax.axvline(0, color='gray', linestyle='--')
        ax.axvline(89, color='red', linestyle='-', alpha=0.3) # Goal Line
        ax.axvline(25, color='blue', linestyle='-', alpha=0.3) # Blue Line
        
        # Plot Vectors
        # Start: Shooter (x, y) -> End: Block (block_x, block_y)
        # We use arrows or segments
        
        # Scatter Points
        ax.scatter(subset['x'], subset['y'], c='blue', label='Shooter (x,y)', alpha=0.6, s=30)
        ax.scatter(subset['block_x'], subset['block_y'], c='red', label='Block (block_x,y)', alpha=0.6, s=30, marker='x')
        
        # Draw Lines
        for idx, row in subset.iterrows():
            ax.plot([row['x'], row['block_x']], [row['y'], row['block_y']], color='gray', alpha=0.5, linewidth=1)
            
            # Optional: Add small arrow
            # ax.arrow(row['x'], row['y'], row['block_x']-row['x'], row['block_y']-row['y'], 
            #          head_width=1, head_length=1, fc='k', ec='k', length_includes_head=True, alpha=0.3)

        ax.set_title(f"Shooting Team ID: {team_name} (n={len(subset)})")
        ax.set_xlabel("Standardized X (ft)")
        ax.set_ylabel("Standardized Y (ft)")
        ax.legend()
        ax.grid(True, alpha=0.2)

    plt.suptitle("Blocked Shot Vectors: Shooter Origin -> Block Location\n(Standardized to Attack Right)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    main()
