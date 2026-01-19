import pandas as pd

df = pd.read_csv('table.csv')
# Remove the 'total' row
df = df[df['rank'] != 'total']
# Calculate medal efficiency (gold / total)
df['medal_efficiency'] = df['gold'].astype(int) / df['total'].astype(int)
# Find the nation with the highest efficiency
max_efficiency_nation = df.loc[df['medal_efficiency'].idxmax(), 'nation']
print(f"Final Answer: {max_efficiency_nation}")