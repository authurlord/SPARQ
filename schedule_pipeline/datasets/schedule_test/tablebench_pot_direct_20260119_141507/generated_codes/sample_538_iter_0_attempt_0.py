import pandas as pd

df = pd.read_csv('table.csv')
# Filter out the total row
df_filtered = df[df['nation'] != 'total']

# Calculate medal efficiency for each nation
df_filtered['efficiency'] = df_filtered['gold'] / df_filtered['total']

# Find the nation with the highest efficiency
max_efficiency_nation = df_filtered.loc[df_filtered['efficiency'].idxmax(), 'nation']
print(f"Final Answer: {max_efficiency_nation}")