import pandas as pd

df = pd.read_csv('table.csv')

# Calculate the percentage of solid waste power relative to total electricity demand
df['solid_waste_percentage'] = (df['solid waste power'] / df['total electricity demand']) * 100

# Find the community with the highest solid waste percentage
max_community = df.loc[df['solid_waste_percentage'].idxmax(), 'autonomous community']

print(f"Final Answer: {max_community}")