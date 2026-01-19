import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row where Silver medals are 30
nation_with_30_silver = df[df['Silver'] == '30']['Nation'].values[0]
print(f"Final Answer: {nation_with_30_silver}")