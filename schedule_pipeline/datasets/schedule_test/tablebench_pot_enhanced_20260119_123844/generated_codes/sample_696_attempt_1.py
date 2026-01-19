import pandas as pd

df = pd.read_csv('table.csv')
# Extract the 'foreign' column and find the max and min values
foreign_column = df['foreign']
max_foreign = foreign_column.max()
min_foreign = foreign_column.min()
max_municipality = df.loc[foreign_column.idxmax(), 'mapiri municipality']  # Use the actual municipality name from the row
min_municipality = df.loc[foreign_column.idxmin(), 'mapiri municipality']

difference = max_foreign - min_foreign
print(f"Final Answer: {max_municipality}, {difference}")