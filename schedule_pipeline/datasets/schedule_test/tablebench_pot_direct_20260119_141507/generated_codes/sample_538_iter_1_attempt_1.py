import pandas as pd

df = pd.read_csv('table.csv')
# Remove the total row
df_filtered = df[df['nation'] != 'total']

# Convert gold and total to numeric
df_filtered['gold'] = pd.to_numeric(df_filtered['gold'], errors='coerce')
df_filtered['total'] = pd.to_numeric(df_filtered['total'], errors='coerce')

# Calculate medal efficiency
df_filtered['efficiency'] = df_filtered['gold'] / df_filtered['total']

# Find the nation with the highest efficiency
max_efficiency_nation = df_filtered.loc[df_filtered['efficiency'].idxmax(), 'nation']
print(f"Final Answer: {max_efficiency_nation}")