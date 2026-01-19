import pandas as pd

df = pd.read_csv('table.csv')
# Find the nation with exactly 30 silver medals
nation_with_30_silver = df[df['Silver'] == 30]['Nation'].values[0]
print(f"Final Answer: {nation_with_30_silver}")