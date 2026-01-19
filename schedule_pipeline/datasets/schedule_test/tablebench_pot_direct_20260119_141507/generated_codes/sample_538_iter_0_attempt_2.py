import pandas as pd

df = pd.read_csv('table.csv')
# Remove the total row (last row)
df = df[df['nation'] != 'total']

# Calculate medal efficiency (gold / total) for each nation
df['efficiency'] = df['gold'] / df['total']

# Find the nation with the highest efficiency
most_efficient_nation = df.loc[df['efficiency'].idxmax(), 'nation']
print(f"Final Answer: {most_efficient_nation}")