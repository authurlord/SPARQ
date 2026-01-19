import pandas as pd

df = pd.read_csv('table.csv')
# Filter for nations with 3 gold and total of 8 medals
result = df[(df['Gold'] == 3) & (df['Total'] == 8)]
# Extract the nation name
nation = result.iloc[0]['Nation'] if not result.empty else None
print(f"Final Answer: {nation}")