import pandas as pd

df = pd.read_csv('table.csv')
# Filter for rank 4 and 4 gold medals
nation = df[(df['Rank'] == '4') & (df['Gold'] == '4')]['Nation'].iloc[0]
print(f"Final Answer: {nation}")