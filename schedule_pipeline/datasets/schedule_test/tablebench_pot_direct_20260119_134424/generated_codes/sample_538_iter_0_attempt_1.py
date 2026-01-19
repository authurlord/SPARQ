import pandas as pd

df = pd.read_csv('table.csv')
# Exclude the 'total' row
df_nations = df[df['rank'] != 'total']
# Calculate medal efficiency
df_nations['medal_efficiency'] = df_nations['gold'].astype(int) / df_nations['total'].astype(int)
# Find the nation with the highest medal efficiency
highest_efficiency_nation = df_nations.loc[df_nations['medal_efficiency'].idxmax(), 'nation']
print(f"Final Answer: {highest_efficiency_nation}")