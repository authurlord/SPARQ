import pandas as pd

df = pd.read_csv('table.csv')
# Filter for nations with total 8 medals and 3 gold medals
filtered_nation = df[(df['Total'] == 8) & (df['Gold'] == 3)]
nation = filtered_nation['Nation'].iloc[0]
print(f"Final Answer: {nation}")