import pandas as pd

df = pd.read_csv('table.csv')
# Filter for nations with Total=8 and Gold=3
filtered_nation = df[(df['Total'] == 8) & (df['Gold'] == 3)]['Nation']
# Output the result
print(f"Final Answer: {filtered_nation.values[0]}")