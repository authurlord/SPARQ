import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'foreign' column to numeric for comparison
foreign_column = df['foreign']
max_foreign_municipality = df.loc[foreign_column.idxmax(), 'mapiri municipality']
min_foreign_municipality = df.loc[foreign_column.idxmin(), 'mapiri municipality']
difference = foreign_column.max() - foreign_column.min()
print(f"Final Answer: {max_foreign_municipality}, {difference}")