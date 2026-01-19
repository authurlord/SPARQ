import pandas as pd

df = pd.read_csv('table.csv')
# Extract the 'foreign' column and find the max and min values with their respective municipalities
foreign_column = df['foreign']
max_municipality = df.loc[foreign_column.idxmax(), 'mapiri municipality']
min_municipality = df.loc[foreign_column.idxmin(), 'mapiri municipality']
max_value = foreign_column.max()
min_value = foreign_column.min()
difference = max_value - min_value

print(f"Final Answer: {max_municipality}, {difference}")