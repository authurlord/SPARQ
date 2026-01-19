import pandas as pd

df = pd.read_csv('table.csv')

# Convert solid waste power and total electricity demand to numeric
df['solid waste power'] = pd.to_numeric(df['solid waste power'], errors='coerce')
df['total electricity demand'] = pd.to_numeric(df['total electricity demand'], errors='coerce')

# Calculate percentage of solid waste power in total electricity demand
df['solid_waste_pct'] = (df['solid waste power'] / df['total electricity demand']) * 100

# Find the row with the maximum percentage
max_row = df.loc[df['solid_waste_pct'].idxmax()]
autonomous_community = max_row['autonomous community']

print(f"Final Answer: {autonomous_community}")