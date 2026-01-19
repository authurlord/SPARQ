import pandas as pd

df = pd.read_csv('table.csv')
# Filter for nations with total = 8 and gold = 3
result = df[(df['Total'] == 8) & (df['Gold'] == 3)]['Nation'].values
print(f"Final Answer: {result[0]}")