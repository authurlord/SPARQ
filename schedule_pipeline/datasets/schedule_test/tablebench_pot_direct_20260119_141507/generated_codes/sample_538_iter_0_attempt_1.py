import pandas as pd

df = pd.read_csv('table.csv')
# Remove the 'total' row
df = df[df['nation'] != 'total']

# Calculate medal efficiency for each nation
df['efficiency'] = df['gold'] / df['total']

# Find the nation with the highest efficiency
highest_efficiency_nation = df.loc[df['efficiency'].idxmax(), 'nation']
print(f"Final Answer: {highest_efficiency_nation}")