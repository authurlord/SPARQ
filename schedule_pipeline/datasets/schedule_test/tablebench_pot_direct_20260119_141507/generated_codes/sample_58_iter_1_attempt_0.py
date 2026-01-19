import pandas as pd

df = pd.read_csv('table.csv')
# Find the nation with 30 silver medals
nation_with_30_silver = df[df['Silver'] == 30]['Nation'].values[0] if df[df['Silver'] == 30].shape[0] > 0 else None
print(f"Final Answer: {nation_with_30_silver}")