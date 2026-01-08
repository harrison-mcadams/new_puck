
import pandas as pd
df = pd.read_csv("data/20252026.csv", nrows=1)
print(df.columns.tolist())
