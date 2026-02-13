import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def main():
    # 1. Load Data
    data_path = Path("analysis/season_shots_20252026.csv")
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows.")

    # 2. Setup Plot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 10))

    # 3. Scatter Plot
    # Downsample for performance if needed, but 100k points is manageable
    # Use alpha for density
    sns.scatterplot(
        data=df, 
        x='xg_nested', 
        y='xtg_mixed', 
        alpha=0.3, 
        s=10,
        edgecolor=None
    )

    # 4. Reference Line (x=y)
    max_val = max(df['xg_nested'].max(), df['xtg_mixed'].max())
    plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', label='Y=X (No Adjustment)')

    # Labels
    plt.title("xG (Base) vs xtG (Mixed Effects)")
    plt.xlabel("Nested xG (Base Model)")
    plt.ylabel("Mixed Effects xtG (Adjusted)")
    plt.legend()

    # 5. Save
    output_dir = Path("analysis/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "xg_vs_xtg_scatter.png"
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    main()
