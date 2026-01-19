import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Silver is 30
nation_with_30_silver = df[df['Silver'] == '30']['Nation'].iloc[0]
print(f"Final Answer: {nation_with_30_silver}")