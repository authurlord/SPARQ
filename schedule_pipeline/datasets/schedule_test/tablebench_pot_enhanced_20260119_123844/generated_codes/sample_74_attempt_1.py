import pandas as pd

df = pd.read_csv('table.csv')
# Filter for nations with total 8 medals and 3 gold medals
filtered_nation = df[(df['Total'] == 8) & (df['Gold'] == 3)]['Nation']
print(f"Final Answer: {filtered_nation.values[0]}")