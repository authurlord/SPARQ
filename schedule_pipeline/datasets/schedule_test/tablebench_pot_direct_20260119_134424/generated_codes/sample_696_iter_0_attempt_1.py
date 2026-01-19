import pandas as pd

df = pd.read_csv('table.csv')
# Extract the 'foreign' column
foreign_column = df['foreign']
# Find the municipality with the highest value
max_municipality = df.loc[foreign_column.idxmax(), 'mapiri municipality']
# Find the municipality with the lowest value
min_municipality = df.loc[foreign_column.idxmin(), 'mapiri municipality']
# Calculate the difference
difference = foreign_column.max() - foreign_column.min()
print(f"Final Answer: {max_municipality}, {difference}")