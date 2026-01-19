import pandas as pd

df = pd.read_csv('table.csv')
# Extract the 'foreign' column and find the municipality with the highest and lowest values
foreign_column = df['foreign']
max_foreign_municipality = df.loc[foreign_column.idxmax(), 'teoponte municipality']
min_foreign_municipality = df.loc[foreign_column.idxmin(), 'teoponte municipality']
difference = foreign_column.max() - foreign_column.min()

print(f"Final Answer: {max_foreign_municipality}, {difference}")