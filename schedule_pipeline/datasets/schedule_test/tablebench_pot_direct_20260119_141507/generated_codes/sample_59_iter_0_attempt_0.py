import pandas as pd

df = pd.read_csv('table.csv')
# Find the row where rank is 4 (index 3)
row = df[df['Rank'] == '4']
nation = row['Nation'].values[0]
print(f"Final Answer: {nation}")