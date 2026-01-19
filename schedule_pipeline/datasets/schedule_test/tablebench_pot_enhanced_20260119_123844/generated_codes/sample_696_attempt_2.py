import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'foreign' column to numeric for comparison
foreign_column = df['foreign'].astype(float)

# Find municipality with highest and lowest foreign speakers
max_municipality = df.loc[foreign_column.idxmax(), 'mapiri municipality']
min_municipality = df.loc[foreign_column.idxmin(), 'mapiri municipality']

# Calculate the difference
difference = foreign_column.max() - foreign_column.min()

print(f"Final Answer: {max_municipality}, {difference:.0f}")