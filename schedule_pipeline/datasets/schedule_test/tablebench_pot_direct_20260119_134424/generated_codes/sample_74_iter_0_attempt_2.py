import pandas as pd

df = pd.read_csv('table.csv')
# Filter for nation with Total=8 and Gold=3
result = df[(df['Total'] == 8) & (df['Gold'] == 3)]
nation = result['Nation'].iloc[0]
print(f"Final Answer: {nation}")