import pandas as pd

df = pd.read_csv('table.csv')
# Remove the total row
df_filtered = df[df['nation'] != 'total']

# Calculate medal efficiency: gold / total
df_filtered['efficiency'] = df_filtered['gold'] / df_filtered['total']

# Find the nation with the highest efficiency
highest_efficiency_nation = df_filtered.loc[df_filtered['efficiency'].idxmax(), 'nation']
print(f"Final Answer: {highest_efficiency_nation}")