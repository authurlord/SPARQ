import pandas as pd

df = pd.read_csv('table.csv')
# Extract the 'foreign' column and find max and min values
foreign_column = df['foreign']
max_foreign = foreign_column.max()
min_foreign = foreign_column.min()
municipality_max = df.loc[foreign_column.idxmax(), 'mapiri municipality']  # Use the municipality name corresponding to max value
municipality_min = df.loc[foreign_column.idxmin(), 'mapiri municipality']  # Use the municipality name corresponding to min value
difference = max_foreign - min_foreign

print(f"Final Answer: {municipality_max}, {difference}")