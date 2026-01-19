import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the percentage of solid waste power in total electricity demand
df['solid_waste_percent'] = (df['solid waste power'] / df['total electricity demand']) * 100
# Find the community with the highest value
max_percent_community = df.loc[df['solid_waste_percent'].idxmax(), 'autonomous community']
print(f"Final Answer: {max_percent_community}")