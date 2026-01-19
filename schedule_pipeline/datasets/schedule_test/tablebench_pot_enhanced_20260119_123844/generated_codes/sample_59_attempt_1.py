import pandas as pd

df = pd.read_csv('table.csv')
# Filter for rank 4 and 4 gold medals
result = df[(df['Rank'] == '4') & (df['Gold'] == '4')]
nation = result['Nation'].values[0]
print(f"Final Answer: {nation}")