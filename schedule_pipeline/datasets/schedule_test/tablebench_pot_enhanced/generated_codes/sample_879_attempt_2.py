import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'solid waste power' and 'total electricity demand' to numeric
df['solid waste power'] = pd.to_numeric(df['solid waste power'], errors='coerce')
df['total electricity demand'] = pd.to_numeric(df['total electricity demand'], errors='coerce')

# Calculate the percentage of solid waste power in total electricity demand
df['% solid waste power'] = (df['solid waste power'] / df['total electricity demand']) * 100

# Find the autonomous community with the highest percentage
max_community = df.loc[df['% solid waste power'].idxmax(), 'autonomous community']

print(f"Final Answer: {max_community}")