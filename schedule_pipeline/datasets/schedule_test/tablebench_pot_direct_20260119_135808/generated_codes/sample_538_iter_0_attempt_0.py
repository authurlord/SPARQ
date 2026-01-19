import pandas as pd

df = pd.read_csv('table.csv')
# Exclude the 'total' row
df_nations = df[df['rank'] != 'total']
# Convert gold and total columns to numeric
df_nations['gold'] = pd.to_numeric(df_nations['gold'])
df_nations['total'] = pd.to_numeric(df_nations['total'])
# Calculate medal efficiency
df_nations['medal_efficiency'] = df_nations['gold'] / df_nations['total']
# Find the nation with the highest efficiency
max_efficiency_nation = df_nations.loc[df_nations['medal_efficiency'].idxmax(), 'nation']
print(f"Final Answer: {max_efficiency_nation}")